"""Physical acceptance using Gazebo feedback, never driving model poses directly."""
from pathlib import Path
import time,json,math,hashlib,sys
import numpy as np,cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image,CameraInfo,LaserScan
from rosgraph_msgs.msg import Clock

OUT=Path(__file__).resolve().parent/'evidence'/time.strftime('acceptance-%Y%m%d-%H%M%S')
OUT.mkdir(parents=True)
class Verify(Node):
    def __init__(self):
        super().__init__('physical_acceptance');self.t=0.;self.rows=[];self.states={};self.images={};self.infos={};self.scans=[];self.phase='settle';self.image_times=[]
        self.pubs={i:self.create_publisher(Twist,f'/assessment/car{i}/cmd_vel',10) for i in range(1,4)}
        self.create_subscription(Clock,'/clock',lambda m:setattr(self,'t',m.clock.sec+m.clock.nanosec*1e-9),qos_profile_sensor_data)
        self.create_subscription(ModelStates,'/assessment/model_states',self.state,10)
        for name,topic in [('main','/assessment/camera/image_raw'),('overview','/assessment/overview/overview/image_raw')]:
            self.create_subscription(Image,topic,lambda m,k=name:self.image(m,k),qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/assessment/camera/camera_info',lambda m:self.infos.update(main=m),qos_profile_sensor_data)
        self.create_subscription(LaserScan,'/assessment/lidar/scan',lambda m:self.scans.append(m),qos_profile_sensor_data)
    def image(self,m,k):
        self.images[k]=m
        if k=='main':self.image_times.append(m.header.stamp.sec+m.header.stamp.nanosec*1e-9)
    def state(self,m):
        for i in range(1,4):
            name=f'astra_car{i}'
            if name not in m.name:continue
            n=m.name.index(name);p=m.pose[n];v=m.twist[n]
            q=p.orientation;y=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            row={'t':self.t,'wall':time.monotonic(),'phase':self.phase,'car':i,'xyz':[p.position.x,p.position.y,p.position.z],'yaw':y,'velocity':[v.linear.x,v.linear.y,v.linear.z]}
            self.states[i]=row;self.rows.append(row)
    def drive(self,commands):
        for i,p in self.pubs.items():
            m=Twist();m.linear.x,m.angular.z=commands.get(i,(0.,0.));p.publish(m)
    def wait(self,seconds,commands=None):
        start=self.t;deadline=time.monotonic()+max(30,seconds*12)
        while self.t-start<seconds and time.monotonic()<deadline:
            self.drive(commands or {});rclpy.spin_once(self,timeout_sec=.025)
        if self.t-start<seconds:raise TimeoutError('Simulation clock did not progress')
    def save_image(self,label):
        for k,m in self.images.items():
            a=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.step)[:,:m.width*3].reshape(m.height,m.width,3)
            if m.encoding=='rgb8':a=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(OUT/f'{label}-{k}.png'),a)
        print(label,self.t,flush=True)

def main():
    rclpy.init();n=Verify();summary={}
    try:
        deadline=time.monotonic()+180
        while (len(n.states)<3 or len(n.images)<2 or not n.infos or not n.scans) and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.1)
        assert len(n.states)==3 and n.infos and n.images and n.scans,'Missing robot/sensor data'
        n.wait(2);n.save_image('initial');initial=dict(n.states);results=[]
        for i in range(1,4):
            n.phase=f'car{i}_only';before=dict(n.states);n.wait(3,{i:(.12,0.)});n.wait(1);after=dict(n.states)
            moves={str(j):float(np.linalg.norm(np.array(after[j]['xyz'])[:2]-np.array(before[j]['xyz'])[:2])) for j in range(1,4)}
            results.append({'controlled_car':i,'displacement_m':moves,'pass':moves[str(i)]>.18 and all(moves[str(j)]<.08 for j in range(1,4) if i!=j)})
            n.save_image(f'car{i}_only')
        n.phase='simultaneous';before=dict(n.states);n.wait(3,{1:(.10,0.),2:(.12,0.),3:(.08,0.)});n.wait(1)
        simultaneous={str(i):float(np.linalg.norm(np.array(n.states[i]['xyz'])[:2]-np.array(before[i]['xyz'])[:2])) for i in range(1,4)}
        n.save_image('simultaneous')
        turns=[]
        for i in range(1,4):
            n.phase=f'turn{i}';before=dict(n.states);n.wait(2,{i:(0.,.2)});n.wait(1)
            delta=math.atan2(math.sin(n.states[i]['yaw']-before[i]['yaw']),math.cos(n.states[i]['yaw']-before[i]['yaw']))
            turns.append({'car':i,'yaw_change_rad':delta,'pass':abs(delta)>.2})
        n.phase='stop';n.wait(2);n.save_image('final')
        info=n.infos['main'];scan=n.scans[-1];im=n.images['main']
        fov=2*math.atan(info.width/(2*info.k[0]))
        sensors={'camera_width':info.width,'camera_height':info.height,'encoding':im.encoding,'K':list(info.k),'D':list(info.d),'horizontal_fov_deg':math.degrees(fov),
                 'laser_count':len(scan.ranges),'laser_fov_deg':math.degrees(scan.angle_max-scan.angle_min),'laser_range_min':scan.range_min,'laser_range_max':scan.range_max,
                 'laser_scan_time':scan.scan_time,'laser_dt_median':float(np.median(np.diff([s.header.stamp.sec+s.header.stamp.nanosec*1e-9 for s in n.scans]))),
                 'image_dt_median':float(np.median(np.diff(n.image_times)))}
        smooth=[]
        for i in range(1,4):
            rows=[r for r in n.rows if r['car']==i and r['t']>0];dt=np.diff([r['t'] for r in rows]);d=np.linalg.norm(np.diff([r['xyz'] for r in rows],axis=0),axis=1)
            # ModelStates has no header. The 10 Hz /clock callback is unsuitable
            # for differentiating 30 Hz poses; use reported physical velocity.
            speed=np.linalg.norm([r['velocity'] for r in rows],axis=1)
            smooth.append({'car':i,'samples':len(rows),'max_step_m':float(d.max()),'max_physics_speed_mps':float(speed.max()),'max_observed_clock_interval_s':float(dt.max()),'pose_header_absent':True,'pass':bool(d.max()<.03 and speed.max()<.4)})
        positions=np.array([initial[i]['xyz'] for i in range(1,4)])
        distances=[float(np.linalg.norm(positions[i,:2]-positions[j,:2])) for i in range(3) for j in range(i+1,3)]
        sensor_pass=info.width==640 and info.height==480 and im.encoding=='rgb8' and abs(math.degrees(fov)-60)<.01 and len(scan.ranges)==50 and abs(sensors['laser_fov_deg']-70.4)<.01 and abs(scan.range_min-.05)<1e-6 and scan.range_max==90 and abs(sensors['laser_dt_median']-.1)<.02
        summary={'independent_commands':results,'simultaneous_displacement_m':simultaneous,'turns':turns,'smoothness':smooth,'sensors':sensors,
                 'initial_positions':positions.tolist(),'initial_min_center_distance_m':min(distances),'separation_pass':min(distances)>1.,'sensor_pass':sensor_pass,
                 'all_pass':all(r['pass'] for r in results+turns+smooth) and all(v>.15 for v in simultaneous.values()) and sensor_pass and min(distances)>1.,
                 'scope':'actual Gazebo diff-drive physics, command topic only; no pose teleport; sampled motion evidence, not an exhaustive collision proof'}
        (OUT/'RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    finally:
        if rclpy.ok():n.drive({})
        (OUT/'states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in n.rows))
        n.destroy_node();rclpy.try_shutdown()
    return 0 if summary.get('all_pass') else 1
if __name__=='__main__':sys.exit(main())
