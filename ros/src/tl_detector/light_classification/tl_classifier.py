from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self, path_to_frozen_path):
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    @staticmethod
    def _load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #image = self._load_image_into_numpy_array(image)
        with self._detection_graph.as_default():
            with tf.Session() as sess:
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = { output.name for op in ops for output in op.outputs }
                tensor_dict = {}
                # get all tensors graph
                for key in [
                    'num_detections', 'detection_boxes',
                    'detection_scores', 'detection_classes',
                    'detection_masks'
                ]:
                    tensor_name = key + ":0"
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
#                rospy.logwarn(tensor_dict)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        scores = output_dict['detection_scores'][0]
        rospy.logwarn(scores)
        idx = np.where(np.array(scores[0:3]) > 0.10)
        if len(idx) == 0:
            idx = -1
        else:
            idx = int(output_dict['detection_classes'][idx[0][0]][0])
        #rospy.logwarn(idx)
        return {1: TrafficLight.RED, 2: TrafficLight.GREEN, 3: TrafficLight.YELLOW, -1: TrafficLight.UNKNOWN}[idx]

