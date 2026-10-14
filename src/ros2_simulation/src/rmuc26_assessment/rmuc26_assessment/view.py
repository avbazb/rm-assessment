from pathlib import Path
import math,xml.etree.ElementTree as E
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker,MarkerArray

def quaternion(r,p,y):
    cr,sr,cp,sp,cy,sy=math.cos(r/2),math.sin(r/2),math.cos(p/2),math.sin(p/2),math.cos(y/2),math.sin(y/2)
    return (sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy)

class View(Node):
    def __init__(self):
        super().__init__('assessment_view')
        model=Path(self.declare_parameter('model_path','').value)
        self.map_pub=self.create_publisher(MarkerArray,'/assessment/map_markers',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.car_pub=self.create_publisher(MarkerArray,'/assessment/robot_markers',10)
        self.map=MarkerArray()
        for i,link in enumerate(E.parse(model/'model.sdf').findall('.//link')):
            v=link.find('visual');m=Marker();m.header.frame_id='world';m.ns='RMUC26_CAD';m.id=i;m.type=Marker.MESH_RESOURCE;m.action=Marker.ADD
            vals=[float(n) for n in link.findtext('pose','0 0 0 0 0 0').split()]
            m.pose.position.x,m.pose.position.y,m.pose.position.z=vals[:3]
            m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w=quaternion(*vals[3:])
            m.scale.x=m.scale.y=m.scale.z=1.
            m.mesh_resource=(model.parent/v.findtext('geometry/mesh/uri').removeprefix('model://')).as_uri()
            m.mesh_use_embedded_materials=True
            self.map.markers.append(m)
        self.create_subscription(ModelStates,'/assessment/model_states',self.state,10)
        self.create_timer(2.,lambda:self.map_pub.publish(self.map))
    def state(self,msg):
        arr=MarkerArray()
        for i in range(1,4):
            name=f'astra_car{i}'
            if name not in msg.name:continue
            pose=msg.pose[msg.name.index(name)]
            m=Marker();m.header.frame_id='world';m.ns='cars';m.id=i;m.type=Marker.CUBE;m.pose=pose
            m.pose.position.z+=.12;m.scale.x=.6;m.scale.y=.48;m.scale.z=.24
            m.color.a=1.;m.color.r=.8 if i!=2 else .05;m.color.g=.12;m.color.b=.8 if i==2 else .12
            arr.markers.append(m)
            label=Marker();label.header.frame_id='world';label.ns='IDs';label.id=i;label.type=Marker.TEXT_VIEW_FACING
            label.pose.position.x=pose.position.x;label.pose.position.y=pose.position.y;label.pose.position.z=pose.position.z+.8
            label.pose.orientation.w=1.;label.scale.z=.3;label.color.a=1.;label.color.r=label.color.g=label.color.b=1.;label.text=f'car{i}'
            arr.markers.append(label)
        self.car_pub.publish(arr)

def main():
    rclpy.init();n=View()
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    finally:n.destroy_node();rclpy.try_shutdown()
