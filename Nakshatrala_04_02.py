
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import scipy.misc
import os
import mpmath
import tensorflow as tf
import itertools

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class DisplayActivationFunctions:

    def __init__(self, root, master, master1, *args, **kwargs):

        self.master = master
        self.root = root
        self.master1 = master1

        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 100
        #self.xasis = np.arange(1000)

        # Importing Mnist data
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.learning_rate = 0.5
        self.lambda_value = 0.01
        self.no_of_nodes = 100
        self.batch_size = 64
        self.percent_data = 10
        self.hidden_activation_function = "Relu"
        self.output_activation_function = "Softmax"
        self.cost_function = "Cross Entropy"
        self.train_set = self.generate_train_set()
        self.test_set = self.generate_test_set()
        self.test_set_target = self.mnist.test.labels
        self.cost_array =[]
        self.error_array=[]
        self.batch_array=[]
        self.batch_c = 0

        self.train_matrix = tf.placeholder(tf.float32, shape=(None, 784))
        self.train_target = tf.placeholder(tf.float32, shape=(None, 10))
        self.test_matrix = tf.constant(self.mnist.test.images)
        self.test_target = tf.constant(self.mnist.test.labels)
        self.hidden_weight_matrix = tf.Variable(tf.truncated_normal([784, self.no_of_nodes]))
        self.hidden_bias_matrix = tf.Variable(tf.truncated_normal([self.no_of_nodes]))
        self.output_weight_matrix = tf.Variable(tf.truncated_normal([self.no_of_nodes, 10]))
        self.output_bias_matrix = tf.Variable(tf.truncated_normal([10]))

        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)
        self.plot_frame.columnconfigure(2, weight=1)
        self.figure = plt.figure("")
        self.p1 = self.figure.add_subplot(1, 2, 1)
        self.p1.set_xlabel("Batch")
        self.p1.set_ylabel("Error Percentage")
        self.p1.set_xlim(self.xmin, self.xmax)
        self.p1.set_ylim(self.ymin, self.ymax)
        self.axes1 = self.figure.gca()
        self.p2 = self.figure.add_subplot(1, 2, 2)
        self.p2.set_xlim(self.xmin, self.xmax)
        self.p2.set_ylim(self.ymin, self.ymax)
        self.p2.set_xlabel("Batch ")
        self.p2.set_ylabel("Cost Percentage")
        self.axes2 = self.figure.gca()
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        ############################

        self.plot_frame2 = tk.Frame(self.master1)
        self.plot_frame2.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame2.rowconfigure(1, weight=1)
        self.plot_frame2.columnconfigure(1, weight=1)
        self.figure2 = plt.figure()
        self.subplt = self.figure2.add_subplot(1,1,1)
        self.subplt.set_aspect(1)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame2)
        self.plot_widget2 = self.canvas2.get_tk_widget()
        self.plot_widget2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)



        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        # settting learning rate slider

        self.learning_rate_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Alpha",
                                             command=lambda event: self.learning_rate_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_callback())
        self.learning_rate_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Lamda slider
        self.lambda_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0.00, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                      activebackground="#FF0000",
                                      highlightcolor="#00FFFF",
                                      label="Lambda",
                                      command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_value)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Number of nodes slider

        self.no_of_nodes_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                           from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                           activebackground="#FF0000",
                                           highlightcolor="#00FFFF",
                                           label="Number of Nodes",
                                           command=lambda event: self.no_of_nodes_slider_callback())
        self.no_of_nodes_slider.set(self.no_of_nodes)
        self.no_of_nodes_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_nodes_slider_callback())
        self.no_of_nodes_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Batch size slider

        self.batch_size_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                          from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Batch Size",
                                          command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        # Percent data slider

        self.percent_data_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Percent data",
                                            command=lambda event: self.percent_data_slider_callback())
        self.percent_data_slider.set(self.percent_data)
        self.percent_data_slider.bind("<ButtonRelease-1>", lambda event: self.percent_data_slider_callback())
        self.percent_data_slider.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)


        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(5, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.buttons_frame.columnconfigure(1, weight=1, uniform='xx')
        # setting Adjust weights button

        self.adjust_weights_button = tk.Button(self.buttons_frame, text="Adjust Weights", command=self.adjust_weights)
        self.adjust_weights_button.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # setting Reset weights button

        self.reset_weights = tk.Button(self.buttons_frame, text="Reset Weights", command=self.reset_weights_callback)
        self.reset_weights.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #Hidden layer transfer function
        self.label_for_hidden_activation_function = tk.Label(self.buttons_frame, text="Hidden layer Activation Function",
                                                      justify="center")
        self.label_for_hidden_activation_function.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.hidden_activation_function_variable = tk.StringVar()
        self.hidden_activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.hidden_activation_function_variable,
                                                          "Relu","Sigmoid",
                                                          command=lambda
                                                              event: self.hidden_activation_function_dropdown_callback())
        self.hidden_activation_function_variable.set("Relu")
        self.hidden_activation_function_dropdown.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Dropdown for Output layer transfer function

        self.label_for_output_activation_function = tk.Label(self.buttons_frame, text="Output Activation function",
                                                      justify="center")
        self.label_for_output_activation_function.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.output_activation_function_variable = tk.StringVar()
        self.output_activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.output_activation_function_variable,
                                                          "Softmax", "Sigmoid",
                                                          command=lambda
                                                              event: self.output_activation_function_dropdown_callback())
        self.output_activation_function_variable.set("Softmax")
        self.output_activation_function_dropdown.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #Cost function dropdwon

        self.label_for_cost_function = tk.Label(self.buttons_frame, text="Cost function",
                                                             justify="center")
        self.label_for_cost_function.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.cost_function_variable = tk.StringVar()
        self.cost_function_dropdown = tk.OptionMenu(self.buttons_frame,
                                                                 self.cost_function_variable,
                                                                 "Cross Entropy", "MSE",
                                                                 command=lambda
                                                                     event: self.cost_function_dropdown_callback())
        self.cost_function_variable.set("Cross Entropy")
        self.cost_function_dropdown.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)


        ############################################################



        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def display_activation_function(self,x, y):
        self.axes.plot(x, y, color='blue')
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.axes.xaxis.set_visible(True)
        self.axes.yaxis.set_visible(True)
        plt.title(self.learning_method + " " +self.activation_function)
        self.canvas.draw()

    ##########################################
    #Citation: The following logic is referred from "https://www.youtube.com/watch?v=fQ8q8LTMzwo&feature=youtu.be"
    ##########################################

    def adjust_weights(self):
        self.figure2.clear()


        train_output = self.output(self.train_matrix)
        if (self.cost_function =="Cross Entropy"):

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_output, labels=self.train_target))
            r_loss = tf.nn.l2_loss(self.hidden_weight_matrix) + tf.nn.l2_loss(self.output_weight_matrix)
            t_loss = loss + (self.lambda_value * r_loss)
            training = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(t_loss)

        elif (self.cost_function =="MSE"):
            loss = tf.reduce_mean(tf.squared_difference(tf.nn.softmax(train_output), tf.nn.softmax(self.train_target)))
            r_loss = tf.nn.l2_loss(self.hidden_weight_matrix) + tf.nn.l2_loss(self.output_weight_matrix)
            t_loss = loss + (self.lambda_value * r_loss)
            training = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(t_loss)

        self.test_output = tf.nn.softmax(self.output(self.test_matrix))
        softmax_train_output = tf.nn.softmax(train_output)

        for i in range(0, 10):
            batch_count = int(60000/self.train_set_count)

            self.batch_c = self.batch_c + 1
            for j in range(0, batch_count):
                batch_data,batch_target = self.mnist.train.next_batch(self.batch_size)
                self.feed_dict = {self.train_matrix : batch_data, self.train_target : batch_target}
                _ = self.session.run(training, feed_dict=self.feed_dict)

                  
            cost_value = self.session.run(t_loss, feed_dict=self.feed_dict)
            self.cost_array.append(cost_value/10)
            p = self.session.run(self.test_output)
            error_value = self.error(p, self.mnist.test.labels)
            self.error_array.append(error_value)
            self.batch_array.append(self.batch_c)
          

        cnf_matrix = confusion_matrix(np.argmax(p, 1), np.argmax(self.mnist.test.labels, 1))
        np.set_printoptions(precision=2)

        self.plot_confusion_matrix(cnf_matrix, classes=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), normalize=False, title="Confusion Matrix")
        
        self.p1.plot(self.batch_array, self.error_array)
        self.p2.plot(self.batch_array, self.cost_array)
        self.canvas.draw()
    ###
    #citation: The following Confusion Matrix code is referred from "http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"
    ##

    def plot_confusion_matrix(self, cm, classes, normalize, cmap=plt.cm.Blues, title='confusin Matrix'):
        """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        tick_marks = np.arange(len(classes))

        self.subplt.set_xlabel("Predicted label")
        self.subplt.set_ylabel("True Label")
        self.subplt.set_title("Confusion Matrix")
        self.subplt.set_xticks(tick_marks,classes)
        self.subplt.set_yticks(tick_marks,classes)

        self.canvas2.show()

    def test_loss(self):
        if (self.cost_function =="Cross Entropy"):

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.test_output, labels=self.mnist.test.labels))
            regulatedloss = tf.nn.l2_loss(self.hidden_weight_matrix) + tf.nn.l2_loss(self.output_weight_matrix)
            totalloss = loss + (self.lambda_value * r_loss)


        elif (self.cost_function =="MSE"):
            loss = tf.reduce_mean(tf.squared_difference(tf.nn.softmax(self.test_output), tf.nn.softmax(self.test_target)))
            regulatedloss = tf.nn.l2_loss(self.hidden_weight_matrix) + tf.nn.l2_loss(self.output_weight_matrix)
            totalloss = loss + (self.lambda_value * r_loss)

        return t_loss

    def output(self,input):
        if self.hidden_activation_function=="Relu":
            self.hidden_output_matrix = tf.nn.relu(tf.matmul(input,self.hidden_weight_matrix)+ self.hidden_bias_matrix)
            if self.output_activation_function=="Softmax":
                self.actual_output_matrix = tf.nn.softmax(tf.matmul(self.hidden_output_matrix, self.output_weight_matrix)+self.output_bias_matrix)
            elif self.output_activation_function=="Sigmoid":
                self.actual_output_matrix = tf.nn.sigmoid(
                    tf.matmul(self.hidden_output_matrix, self.output_weight_matrix) + self.output_bias_matrix)

        elif self.hidden_activation_function=="Sigmoid":
            self.hidden_output_matrix = tf.nn.sigmoid(tf.matmul(input,self.hidden_weight_matrix)+ self.hidden_bias_matrix)
            if self.output_activation_function=="Softmax":
                self.actual_output_matrix = tf.nn.softmax(tf.matmul(self.hidden_output_matrix, self.output_weight_matrix)+self.output_bias_matrix)
            elif self.output_activation_function=="Sigmoid":
                self.actual_output_matrix = tf.nn.sigmoid(
                    tf.matmul(self.hidden_output_matrix, self.output_weight_matrix) + self.output_bias_matrix)


        return self.actual_output_matrix

    def hidden_layer_output(self, input):
        if self.hidden_activation_function=="Relu":
            self.hidden_output_matrix = tf.nn.relu(tf.matmul(input,self.hidden_weight_matrix)+ self.hidden_bias_matrix)
        elif self.hidden_activation_function=="Sigmoid":
            self.hidden_output_matrix = tf.nn.sigmoid(tf.matmul(input,self.hidden_weight_matrix)+ self.hidden_bias_matrix)

        return self.hidden_output_matrix

    def actual_output(self):
        if self.output_activation_function=="Softmax":
            self.actual_output_matrix = tf.nn.softmax(tf.matmul(self.hidden_output_matrix,self.output_weight_matrix)+self.output_bias_matrix)
        elif self.output_activation_function=="Sigmoid":
            self.actual_output_matrix = tf.nn.sigmoid(tf.matmul(self.hidden_output_matrix, self.output_weight_matrix) + self.output_bias_matrix)

        return self.actual_output_matrix

    def error(self,output,actual):
        return 100 - 100 * np.sum(np.argmax(output, 1) == np.argmax(actual, 1))/output.shape[0]

    def reset_weights_callback(self):
        tf.global_variables_initializer().run()
        self.cost_array = []
        self.error_array = []
        self.batch_array = []
        self.batch_c = 0
        self.p1.cla()
        self.p2.cla()
        self.p1.set_xlim(self.xmin, self.xmax)
        self.p1.set_ylim(self.ymin, self.ymax)
        self.p2.set_xlim(self.xmin, self.xmax)
        self.p2.set_ylim(self.ymin, self.ymax)
        self.p1.set_xlabel("Batch")
        self.p1.set_ylabel("Error Percentage")

        self.p2.set_xlabel("Batch ")
        self.p2.set_ylabel("Cost Percentage")
        self.canvas.draw()
    def learning_rate_callback(self):
        self.learning_rate = self.learning_rate_slider.get()

    def lambda_slider_callback(self):
        self.lambda_value = self.lambda_slider.get()

    def no_of_nodes_slider_callback(self):
        self.no_of_nodes = self.no_of_nodes_slider.get()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()

    def percent_data_slider_callback(self):
        self.percent_data = self.percent_data_slider.get()

    def hidden_activation_function_dropdown_callback(self):
        self.hidden_activation_function = self.hidden_activation_function_variable.get()


    def output_activation_function_dropdown_callback(self):
        self.output_activation_function = self.output_activation_function_variable.get()

    def cost_function_dropdown_callback(self):
        self.cost_function = self.cost_function_dropdown.get()

    def generate_train_set(self):
        self.train_set_count = np.around((self.percent_data/100)*60000, decimals=-1)
        self.train_set_without_normalize = self.mnist.train.images[0:int(self.train_set_count),:]
        self.train_set = (self.train_set_without_normalize/127.5)-1
        return self.train_set_without_normalize

    def generate_test_set(self):
        self.test_set_count = np.around((self.percent_data / 100) * 60000, decimals=-1)
        self.test_set_without_normalize = self.mnist.test.images[0:int(self.test_set_count), :]
        self.test_set = (self.test_set_without_normalize / 127.5) - 1
        return self.test_set_without_normalize






