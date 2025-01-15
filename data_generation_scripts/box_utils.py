import numpy as np

def nms(bounding_boxes, confidence_score=None, threshold=0.5):
	# If no bounding boxes, return empty list
	if len(bounding_boxes) == 0:
		return []

	if confidence_score is None:
		confidence_score = [1]*len(bounding_boxes)
	# Bounding boxes
	boxes = np.array(bounding_boxes)

	# coordinates of bounding boxes
	start_x = boxes[:, 0]
	start_y = boxes[:, 1]
	end_x = boxes[:, 2]
	end_y = boxes[:, 3]

	# Confidence scores of bounding boxes
	score = np.array(confidence_score)

	# Picked bounding boxes
	picked_boxes = []
	picked_score = []

	# Compute areas of bounding boxes
	areas = (end_x - start_x + 1) * (end_y - start_y + 1)

	# Sort by confidence score of bounding boxes
	order = np.argsort(score)

	# Iterate bounding boxes
	while order.size > 0:
	# The index of largest confidence score
		index = order[-1]

		# Pick the bounding box with largest confidence score
		picked_boxes.append(bounding_boxes[index])
		picked_score.append(confidence_score[index])

		# Compute ordinates of intersection-over-union(IOU)
		x1 = np.maximum(start_x[index], start_x[order[:-1]])
		x2 = np.minimum(end_x[index], end_x[order[:-1]])
		y1 = np.maximum(start_y[index], start_y[order[:-1]])
		y2 = np.minimum(end_y[index], end_y[order[:-1]])
		
		w = np.maximum(0.0, x2 - x1 + 1)
		h = np.maximum(0.0, y2 - y1 + 1)
		intersection = w * h

		# Compute the ratio between intersection and union
		ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

		left = np.where(ratio < threshold)
		order = order[left]

	return picked_boxes

def iou_calc(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	if area_union > 0:
		iou = area_inter / area_union
		return iou
	else:
		return 0

def area_calc(box):
	x1, y1, x2, y2 = box
	return (x2-x1)*(y2-y1)

def remove_outlier(bboxes, criteria='iou', threshold=0.7):
	if len(bboxes) > 4:
		if criteria=='iou':
			mean_x1 = sum([box[0] for box in bboxes]) / len(bboxes)
			mean_y1 = sum([box[1] for box in bboxes]) / len(bboxes)
			mean_x2 = sum([box[2] for box in bboxes]) / len(bboxes)
			mean_y2 = sum([box[3] for box in bboxes]) / len(bboxes)
			mean_box = [mean_x1, mean_y1, mean_x2, mean_y2]
			return [box for box in bboxes if iou_calc(mean_box, box) > threshold]
		elif criteria=='area':
			median_area = np.median([area_calc(box) for box in bboxes])
			return [box for box in bboxes if abs(area_calc(box) - median_area)/median_area < threshold]
	else:
		return bboxes


# def nms(boxes, iou_threshold=0.4):
#     bbox_list_new = []
#     current_box = boxes.pop(0)
#     bbox_list_new.append(current_box)
#     for box in boxes:
#         iou = iou_calc(current_box, box)
#         if iou > iou_threshold:
#             boxes.remove(box)
#     return bbox_list_new