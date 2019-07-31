'''
This module provides consistent params loading from rosparam and command line input
'''

import argparse
import rospy
from easydict import EasyDict as edict

# TODO: create a fake ArgumentParser to show additional information and support complex arguments
def parse_private_args(**kvargs):
    parser = argparse.ArgumentParser()
    node_name = rospy.get_name()

    ros_args = {}
    for name, default in kvargs.items():
        parser.add_argument('--' + name, default=default, type=type(default))
        ros_args[name] = rospy.get_param(node_name + '/' + name, default)
    
    # ------------ ros param system conflict with argparse syntax
    # cmd_args = parser.parse_args()
    # for k, v in vars(cmd_args).items():
    #     if k in ros_args and v != ros_args[v]:
    #         rospy.logwarn("Duplicate arguments {}: {} / {}".format(k, ros_args[k], v))
    #     if v != kvargs[k]:
    #         ros_args[k] = v
    return edict(ros_args)
