"""
This file is only used for transformating the QuickDraw dataset in a Numpy more convenient dataset,
in order to make the fitting as fastest as possible
"""

import ndjson
import os
import numpy as np
import time
objective_dataset = 'NumpyQuickDrawDataset'
original_dataset = 'QuickdrawDataset'
recognized_instances_to_read = 60000+10000
non_recognized_instances_to_read = 5000
modify_stroke = True


def get_original_stroke(drawing):
    """
    Get the stroke as come in the dataset
    :param drawing: Stroke defined at the ndjson file
    :return:
    Numpy stroke with the same format as the ndjson
    """
    list_of_strokes = [np.array(stroke, dtype=np.int32) for stroke in drawing]
    array_of_strokes = np.empty(len(list_of_strokes), dtype=np.object)
    array_of_strokes[:] = list_of_strokes
    return array_of_strokes

def get_modified_stroke(drawing):
    """
    Get all the strokes of a drawing in a unique array, with third channel defining if this point
    is where the stroke starts or a continuation.
    :param drawing: Stroke defined at the ndjson file
    :return:
    Numpy stroke with a flatten format (used for feeding a recurrent network)
    """
    continuation_id, starting_id = 1, 2
    #append 2 for starting on draw 1 for continuation
    [stroke.append([starting_id if i == 0 else continuation_id for i in range(len(stroke[0]))]) for stroke in drawing]

    continuous_array_of_strokes = np.concatenate(drawing, axis=-1).T
    return continuous_array_of_strokes.astype(np.float32)

if __name__ == '__main__':
    for label in os.listdir(original_dataset):
        original_class_path = os.path.join(original_dataset, label)
        label = label[:-len('.ndjson')]
        objective_class_recognizeds_path = os.path.join(objective_dataset, 'Recognizeds')
        objective_class_non_recognizeds_path = os.path.join(objective_dataset, 'NonRecognizeds')
        if not os.path.isdir(objective_class_recognizeds_path):
            os.makedirs(objective_class_recognizeds_path)
        if not  os.path.isdir(objective_class_non_recognizeds_path):
            os.makedirs(objective_class_non_recognizeds_path)
        recognized_instances = []
        non_recognized_instances = []
        start = time.time()
        with open(original_class_path) as f:
            reader = ndjson.reader(f)
            for post in reader:
                if post['recognized']:
                    if len(recognized_instances) >= recognized_instances_to_read:
                        continue
                    list_to_append = recognized_instances
                else:
                    if len(non_recognized_instances) >= non_recognized_instances_to_read:
                        continue
                    else:
                        list_to_append = non_recognized_instances
                if modify_stroke:
                    array_of_strokes = get_modified_stroke(drawing=post['drawing'])
                else:
                    array_of_strokes = get_original_stroke(drawing=post['drawing'])
                list_to_append.append(array_of_strokes)
                if len(recognized_instances) >= recognized_instances_to_read and len(non_recognized_instances) >= non_recognized_instances_to_read:
                    break
            else:
                if len(recognized_instances) < recognized_instances_to_read:
                    raise ValueError("Class "+label+" only have "+str(len(recognized_instances))+" recognized instances.")

        np.save(file=os.path.join(os.path.join(objective_class_non_recognizeds_path, label)),
                arr=np.array(non_recognized_instances), fix_imports=False)
        np.save(file=os.path.join(os.path.join(objective_class_recognizeds_path, label)),
                arr=np.array(recognized_instances), fix_imports=False)

        print("Class "+label.title()+". Generated in "+str((time.time()-start))+" seconds")


