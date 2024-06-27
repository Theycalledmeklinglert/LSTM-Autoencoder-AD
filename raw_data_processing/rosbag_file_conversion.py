import csv
import traceback

import bagpy
import rosbag
from bagpy import bagreader
import genpy
import os


# Sample time of approximately 10 milliseconds for wheelspeed
def read_File(fName):
    # rosbag = bagreader(fName)
    # print(rosbag.topic_table)
    # rosbag.get

    bag = rosbag.Bag(fName)
    topic_info = bag.get_type_and_topic_info()[1]
    topics = list(topic_info.keys())
    types = [val[0] for val in topic_info.values()]

    print("Types:", types)
    print("Topics:", topics)


    #i = 0
    processed_topics = {topic: False for topic in topics}

    #for topic, msg, t in bag.read_messages(topics=['/can_interface/wheelspeed']):
    try:
        for topic, msg, t in bag.read_messages(topics=topics):

            #print(msg)

            #if i > 9:
            #    break
            #print(msg.__slots__)

            if not processed_topics[topic]:
                print(f"Processing topic: {topic}")
                print(f"Message slots: {msg.__slots__}")

                msg_data = {}
                for slot_name in msg.__slots__:
                    attr = getattr(msg, slot_name)
                    if isinstance(attr, bytes):
                        try:
                            attr = attr.decode('utf-8')
                        except (SyntaxError, UnicodeError):
                            try:
                                attr = attr.decode('latin1')  # or another encoding
                            except (SyntaxError, UnicodeError) as e:
                                print(f"Failed to decode attribute {slot_name}: {e}")
                                continue
                    msg_data[slot_name] = attr
                print("Message data:", msg_data)

            processed_topics[topic] = True

            # Check if all topics have been processed
            if all(processed_topics.values()):
                break

            #i += 1

    except ((SyntaxError, UnicodeError), genpy.message.MessageException) as e:
            print(f"An error occurred: {e}")

    #print(msg_data.get("header").stamp.secs)
    #print(msg_data.get("FL").data)
    #print(type(msg_data.get("FL").data))

    #print(type(msg_data.get("FL")))

    get_sample_time(bag, '/can_interface/wheelspeed')
    get_sample_time(bag, '/can_interface/current_steering_angle')
    get_sample_time(bag, '/lidar/cone_position_cloud')

    bag.close()


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

def read_bag_to_csv(bag_file, csv_file):
    try:
        bag = rosbag.Bag(bag_file)
    except Exception as e:
        print(f"Failed to open bag file: {e}")
        return

    # Open CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        header_written = False
        try:
            # Iterate over all topics and messages
            for topic, msg, t in bag.read_messages():
                    if not header_written:
                        # Write header based on message fields
                        header = ['topic', 'timestamp'] + msg.__slots__
                        writer.writerow(header)
                        header_written = True

                    # Extract message data
                    row = [topic, t.to_sec()]
                    for slot in msg.__slots__:
                        attr = getattr(msg, slot)
                        if isinstance(attr, (int, float, str, bool)):
                            row.append(attr)
                        elif hasattr(attr, '__slots__'):
                            for sub_slot in attr.__slots__:
                                sub_attr = getattr(attr, sub_slot)
                                if isinstance(sub_attr, (int, float, str, bool)):
                                    row.append(sub_attr)
                                else:
                                    row.append(str(sub_attr))
                        else:
                            row.append(str(attr))

                    # Write message data to CSV
                    writer.writerow(row)

        except (SyntaxError, UnicodeError) as e:
            print(f"An error occurred while processing topic {topic}: {e}")
            print("FUCK V2: NEW AND IMPROVED")
            print(traceback.format_exc())

    bag.close() #redundant i think

def stupid_encoding_error(bag_file):
    # b = bagreader('./aufnahmen/tmp/autocross_valid_16_05_23.bag')
    # csvfiles = []
    # for t in b.topics:
    #     data = b.message_by_topic(t)
    #     csvfiles.append(data)

    # with open(bag_file, 'r', encoding='utf-16') as f:    #errors='ignore'
    #     print(f.read())
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages():
        print(msg)