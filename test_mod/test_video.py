import os
import torch
import time
from core.utils import *
from datasets.meters import AVAMeter

@torch.no_grad()
def test_ava(cfg, epoch, model, test_loader):
    mps_device = torch.device("mps")

    # Test parameters
    num_classes       = cfg.MODEL.NUM_CLASSES
    anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors       = cfg.SOLVER.NUM_ANCHORS
    nms_thresh        = 0.5
    conf_thresh_valid = 0.005

    nbatch = len(test_loader)
    meter = AVAMeter(cfg, cfg.TRAIN.MODE, 'latest_detection.json')

    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        # data = batch['clip'].cuda()
        data = batch['clip'].to(mps_device)
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}

        with torch.no_grad():
            output = model(data)
            metadata = batch['metadata'].cpu().numpy()

            preds = []
            all_boxes = get_region_boxes_ava(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                
                for box in boxes:
                    x1 = float(box[0]-box[2]/2.0)
                    y1 = float(box[1]-box[3]/2.0)
                    x2 = float(box[0]+box[2]/2.0)
                    y2 = float(box[1]+box[3]/2.0)
                    det_conf = float(box[4])
                    cls_out = [det_conf * x.cpu().numpy() for x in box[5]]
                    preds.append([[x1,y1,x2,y2], cls_out, metadata[i][:2].tolist()])

        meter.update_stats(preds)
        logging("[%d/%d]" % (batch_idx, nbatch))

    mAP = meter.evaluate_ava()
    logging("mode: {} -- mAP: {}".format(meter.mode, mAP))

    return mAP


@torch.no_grad()
def test_random(cfg, epoch, model, test_loader):
    mps_device = torch.device("mps")

    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes = cfg.MODEL.NUM_CLASSES
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = cfg.SOLVER.NUM_ANCHORS
    conf_thresh_valid = 0.005
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)

    print("length of test_loader: ", nbatch)

    model.eval()


    for batch_idx, (base_det_folder, frame_idx, data, target) in enumerate(test_loader):
        # data = data.cuda()
        data = data.to(mps_device)

        with torch.no_grad():
            output = model(data).data

            # print(output[:1])

            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

            for i in range(output.size(0)):
                boxes = all_boxes[i]
                # print("init boxes: ", len(boxes))
                boxes = nms(boxes, nms_thresh)
                # print("init nms boxes: ", len(boxes))


                detection_path = os.path.join(base_det_folder[i], frame_idx[i])

                if not os.path.exists(os.path.dirname(detection_path)):
                    os.makedirs(os.path.dirname(detection_path), exist_ok=True)

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = max(round(float(box[0]-box[2]/2.0) * 320.0), 0.0)
                        y1 = max(round(float(box[1]-box[3]/2.0) * 240.0), 0.0)
                        x2 = min(round(float(box[0]+box[2]/2.0) * 320.0), 320.0)
                        y2 = min(round(float(box[1]+box[3]/2.0) * 240.0), 240.0)


                        f_detect.write(
                            str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(box[4].item()) + ' ' +
                            (" ".join([str(elmt.item()) if torch.is_tensor(elmt) else str(elmt) for elmt in box[5:]])) + '\n'
                        )

                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
                pred_list = [] # LIST OF CONFIDENT BOX INDICES

                if (num_gts > 0):
                    for i in range(len(boxes)):
                        if boxes[i][4] > 0.25:
                            proposals = proposals+1
                            pred_list.append(i)

                    for i in range(num_gts):
                        box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        best_iou = 0
                        best_j = -1
                        for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > best_iou:
                                best_j = j
                                best_iou = iou

                        if best_iou > iou_thresh:
                            total_detected += 1
                            if int(boxes[best_j][6]) == box_gt[6]:
                                correct_classification += 1

                        if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                            correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    locolization_recall = 1.0 * total_detected / (total + eps)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Locolization recall: %.3f" % locolization_recall)

    return fscore
