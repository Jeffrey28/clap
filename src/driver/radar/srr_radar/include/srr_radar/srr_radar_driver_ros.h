#ifndef SRR_RADAR_DRIVER_ROS_H_
#define SRR_RADAR_DRIVER_ROS_H_

// #include <cstdlib>
#include <string>
// #include <sstream>
#include <cstdint>

#include "ros/ros.h"
#include "ros/time.h"
#include "std_msgs/String.h"
#include "srr_radar/common/boost_udp.h"
#include "srr_radar/proto/srr_radar.h"
#include "srr_radar/proto/srr_radar_conf.h"
#include "srr_radar/frame/canet/canet.h"

//headers in Autowae Health Checker
//#include <health_checker/node_status_publisher.h>

namespace drivers {
#define UDP_RECV_BUFFER_SIZE  1024
class SrrRadarDriverRos {
public:    
    SrrRadarDriverRos()  {
        // boost_udp_ = nullptr;
    }
    ~SrrRadarDriverRos() {} // { delete boost_udp_;}
    void init(const ros::NodeHandle* parent);
    void run();
    void process();
    void publish();
private:
    std::string frame_id_;
    std::string topic_name_;
    std::string host_ip_;
    std::string target_ip_;
    int host_port_;
    int target_port_;
    ros::Publisher pub_object_;
    ros::Publisher pub_markerarray_;
    ros::Publisher pub_text_;
    ros::NodeHandle* nh_;
    uint8_t buffer_[UDP_RECV_BUFFER_SIZE];
    common::BoostUdp* boost_udp_;
   // std::shared_ptr<health_checker::NodeStatusPublisher> node_status_pub_ptr_;


    drivers::SrrRadar srr_radar_;
    drivers::srr_radar::SrrRadarConf srr_radar_conf_;     ///> DONE: segmentation fault when constructed.
                                                                ///> use SrrRadarConf instead of RadarConf 
                                                                ///> when constructed, it constructs a RadarConf pointer.
                                                                ///> before use of it, new a object for it.
    drivers::canet::CanetFrame canet_;
};

} // end of namespace drivers
#endif // SRR_RADAR_DRIVER_ROS_H_