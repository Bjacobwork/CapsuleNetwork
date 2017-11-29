import CapsuleNetwork as capsNet
import numpy as np
from collections import deque
import color
import time
import datetime
import threading

run_timer = False
def timer():
    global run_timer
    run_timer = True
    start = time.time()
    while run_timer:
        print("\r"+color.MAGENTA+"Process Time: {}".format(str(datetime.timedelta(seconds=time.time()-start)))+color.RESET, end='')
    print()

def train(model, training_iterations, training_batch_size, testing_batch_size, log_iteration, test_iteration, content):
    """
    Purpose:
        To train a given model.
    Precondition:
        :param model: The model to be trained.
        :param training_iterations: Total training iterations to present the model with.
        :param training_batch_size: The size of each training batch.
        :param testing_batch_size: The size of each testing batch.
        :param log_iteration: The number of iterations between each save.
        :param test_iteration: The number of iterations between each test.
        :param content: The data set interface.
    Postcondition:
        The model was trained on the given data set.
    """
    global run_timer
    training_accuracies = deque()
    average_iteration = 10
    save = False
    while model.total_iterations < training_iterations:
        batch_x, batch_y = content.get_training_batch(training_batch_size, True)
        clock = time.time()
        acc = model.train(batch_x, batch_y)
        tick = time.time()
        model.total_time += (tick - clock)
        time_info = str(datetime.timedelta(seconds=model.total_time))
        model.total_iterations += 1

        training_accuracies.append(acc)
        if len(training_accuracies) > average_iteration:
            training_accuracies.popleft()
        acc = np.mean(list(training_accuracies))

        total_passes = model.total_iterations * training_batch_size
        print("\rIteration: {0:>8} | Total Passes {1:>8} | Training Accuracy: {2:>5.2f}% | Time: {3:>10}".format(model.total_iterations, total_passes, (acc * 100), time_info), end='')

        if model.total_iterations % log_iteration == 0:
            save = True
        if save:
            try:
                print(color.MAGENTA + "\nSaving...")
                save_timer = threading.Thread(target=timer)
                save_timer.start()
                model.save()
                run_timer = False
                time.sleep(.1)
                print(color.BRIGHT_MAGENTA + "Saved!" + color.RESET)
            except:
                print(color.RED + "An error has occurred." + color.RESET)
            save = False
        if model.total_iterations % test_iteration == 0:
            batch_x, batch_y = content.get_testing_batch(testing_batch_size, True)
            clock = time.time()
            _, acc = model.metrics(batch_x, batch_y)
            tick = time.time()
            train_time = (tick - clock)
            time_info = str(datetime.timedelta(seconds=train_time))
            info = "Test Results:\nIteration: {0:>8} | Accuracy: {1:>5.2f} | Time: {2:>10}".format(model.total_iterations, (acc*100), time_info)
            print("\n"+color.BLUE+info+color.RESET, end="\n\n")