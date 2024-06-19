import bagpy
import rosbag
from bagpy import bagreader



# Sample time of approximately 10 milliseconds for wheelspeed
def read_File(fName):
    # rosbag = bagreader(fName)
    # print(rosbag.topic_table)
    # rosbag.get

    bag = rosbag.Bag(fName)
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []
    for val in bag.get_type_and_topic_info()[1].values():
        types.append(val[0])

    print(types)
    print(topics)

    i = 0

    for topic, msg, t in bag.read_messages(topics=['/can_interface/wheelspeed']):
        #print(msg)

        if i > 9:
            break

        print(msg.__slots__)

        msg_data = {}
        for slot_name in msg.__slots__:
            msg_data[slot_name] = getattr(msg, slot_name)
            print(str(slot_name) + ": " + str(msg_data[slot_name]))

        i += 1

    #print(msg_data.get("header").stamp.secs)
    print(msg_data.get("FL").data)
    print(type(msg_data.get("FL").data))

    print(type(msg_data.get("FL")))

    get_sample_time(bag, '/can_interface/wheelspeed')
    get_sample_time(bag, '/can_interface/current_steering_angle')
    get_sample_time(bag, '/lidar/cone_position_cloud')

def get_sample_time(bag, topicName):
    i = 0
    time_diffs = []
    prev_timestamp = 0

    for topic, msg, t in bag.read_messages(topics=[topicName]):
        # print(msg)

        if i > 9:
            break

        #print(msg.__slots__)

        # Sample time of approximately 10 milliseconds for wheelspeed
        curr_timestamp = t.secs * 1e9 + t.nsecs
        time_diffs.append(str(curr_timestamp - prev_timestamp))
        prev_timestamp = (t.secs * 1e9 + t.nsecs)
        i += 1

    # t is more accurate
    # print("t: " + str(t))
    # combined_nanoseconds = t.secs * 1e9 + t.nsecs
    # print("Combined secs and nsecs: " + str(combined_nanoseconds))

    print("Sample time differences for " + str(topicName)[str(topicName).rfind("/") + 1:] + ": " + str(time_diffs))
