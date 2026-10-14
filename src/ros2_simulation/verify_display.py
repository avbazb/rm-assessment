"""Read-only evidence that the one-launch RViz subscribes to complete displays."""
from pathlib import Path
import json,time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from visualization_msgs.msg import MarkerArray

rclpy.init();n=Node('assessment_display_audit');received={}
n.create_subscription(MarkerArray,'/assessment/map_markers',lambda m:received.update(map=m),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
n.create_subscription(MarkerArray,'/assessment/robot_markers',lambda m:received.update(robots=m),10)
deadline=time.monotonic()+20
while len(received)<2 and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.1)
topics=['/assessment/map_markers','/assessment/robot_markers','/assessment/camera/image_raw','/assessment/lidar/scan']
subs={t:[{'node':x.node_name,'namespace':x.node_namespace} for x in n.get_subscriptions_info_by_topic(t)] for t in topics}
maps=received.get('map',MarkerArray()).markers;robots=received.get('robots',MarkerArray()).markers
missing=[m.mesh_resource for m in maps if not Path(m.mesh_resource.removeprefix('file://')).is_file()]
ids=sorted(m.id for m in robots if m.ns=='cars')
out={'map_markers':len(maps),'missing_mesh_paths':missing,'robot_ids':ids,'rviz_subscriptions':subs,
     'all_pass':len(maps)==446 and not missing and ids==[1,2,3] and all(any(x['node'] in ('rviz','rviz2') for x in rows) for rows in subs.values()),
     'scope':'ROS messages, file references and RViz subscription proof; not a GUI screenshot'}
output=Path(__file__).resolve().parent/'evidence/display_audit.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(out,indent=2))
print(json.dumps(out));n.destroy_node();rclpy.shutdown()
raise SystemExit(0 if out['all_pass'] else 1)
