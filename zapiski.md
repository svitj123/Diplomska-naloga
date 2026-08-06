Dobro bi bilo da si čim več stvari zabeležim

Moja naloga je bolj rezutlatska, torej je cilj da prvo dobim
rezultate članka, potem pa eksperimenti moji in kaki so rezultati

Izberem eno ali dve spremembi:
- eno cim manjso (recimo PCA)
- potem pa namesto CNN naredim transaformer

OA je lahko problmatiče, log-loss pa gleda verjetnost

Bolje bi bilo, da se fokusiram na log-loss in recimo za primerjavo
zraven OA

Lokalna validacija, da damo en krogec stran in ga uporabimo za test.
Vzamemo en krog in potem naredilo en bufferzone. Če vzamemo en cel
krog recimo za test, potem ni treba bufferja.

Popraviti morem validacijo in mu pokazati tabelco z rezultati.
Poženem potem še enkrat modele in mu pokažem OA in log-loss.

Menda lahko diplomant napiše kar članek namesto old-school diplome.
Menotor in somentor pravita da bi ne bila slaba opcija.

Do nekje sredine Julija končna verzija. Rad bi diplomiral konec
avgusta. Prvo morem narediti dobre rezultate dam to mentorju, potem
pa grem pisat in upam da jim bo koncept všeč.

Mentor pravi, da se velike resulucije ne dotikam. Nizko so dal na
tekmovanje, srednjo naj se jaz dotikam (to je tamanjša iz članka),
večjo pa naj pustim.

Nadgradnjo delam prvo na manjših podatkih, potem pa grem na nove.



-- Branje članka --
The classification resutls identified a SVM using a RBF kernel as the opttimal spectral-based classifier for this data set as it achivede the best overall calassification accuracy. 

2.3 Data pre-processing

The community has established a set of common pre-processing protocols that have been shown to be effective for biological samples. In this work we apply the following pre-processing steps using in-house implemented software:

1. Baseline correction - Scattering through the specimen is mitigated by applying piece-wise linear (rubber band) baseline correction.

2. Normalization - Normalization is performed by dividing the baseline corrected spectra by Amide I absorbance at ≈ 1650 cm⁻¹.

3. Dimensionality reduction - We apply principal component analysis (PCA), keeping 16 principal components, which captures 90.03% and 96.86% of the spectral variance for SD and HD data, respectively.

Each convolutional layer calculates the convolution of the input with a set of filters that are trained to detect particular features. The convolution layer is followed by an element-wise nonlinear activation operation. The convolution and activation layer weights and biases are calculated during training. The activity of the j^th feature map in the l^th layer is computed as:

F_j^l = g( sum_{i=1}^{N_f} ( W_{i,j}^l * F_i^{l-1} + b_j^l ) ),

where N_f denotes the number of feature maps in the (l - 1)^th layer, F_i^{l-1} ∈ R^{m×n} is the i^th feature map in the (l - 1)^th layer that connects to feature map F_j^l in the l^th layer, W_{i,j}^l ∈ R^{k×k} is the convolutional kernel (of size k) for F_i^{l-1}, b_j^l is the bias, g(·) is a nonlinear activation function such as tanh or a rectified linear unit (ReLu), and * denotes the discrete convolution operator.

The convolutional layer is often followed by a pooling layer, with max pooling used as the most common pooling algorithm. Max pooling computes the maximum in a local window of the input feature map. By using a stride larger than 1, this results in subsampling of the input feature map, which in turn reduces the number of parameters and therefore the computational complexity. A 2 × 2 pooling filter is the most common, which reduces the spatial dimensions of the output by half. The pooling layer is used to increase the robustness to small variations in the location of features detected by the convolutional layer.

The last module of a CNN typically consists of several fully connected layers, similar to a traditional ANN. The extracted high-level features are flattened to a fixed-dimensional vector. The feature vector learned by the l-th fully connected layer can be

A convolution and max pooling based set of layers are introduced, followed by fully connected layers. In particular, one convolution layer consisting of 32 feature maps is followed by a max pooling layer with a kernel size of 2 × 2. This reduces the spatial dimension of the images by a factor of 2. The max pooling layer is followed by two additional convolution layers consisting of 64 feature maps each. An additional max pooling layer, with a kernel size of 2 × 2, is introduced followed by a fully connected layer of 128 units. The strides size is fixed as 1. For all convolution layers, we use kernels of size 3 × 3. The network ends with a softmax layer of size equal to the number of classes so that the final output is a vector of class probabilities for each pixel.

2.5.1 Software

All data pre-processing was performed using our open-source SIproc software,45 implemented in C++ and CUDA. Training and testing was performed in Python using open-source software packages. The Scikit-learn package52 was used for traditional classifiers (SVM, Random Forests, etc.) and TensorFlow, leveraging the TFlearn interface,53 was used to design and implement CNNs.

2.5.2 Implementation Hyperparameters

The choice of hyperparameters is crucial when designing a deep learning architecture, significantly influencing overall accuracy and convergence speed. Through extensive experimentation on our training set, we chose the following hyperparameters:

1. Optimization method - We used an Adadelta54 adaptive learning rate method with a learning rate of r = 0.1. Adadelta adapts the learning rate over time, removing the need to manually tune for our application.

2. Regularization of the weights - We used ℓ2 optimization combined with dropout55 to minimize overfitting. Dropout keeps the activation of a fraction of hidden nodes and it randomly turns off the activation of the rest of the nodes in the layer based on a keep probability threshold. The keep probability is set to 0.5 and 1 in training and testing modes, respectively.

3. Batch normalization - During training, the distribution of each CNN layer changes as the parameters of the previous layers change. This shift of the hidden unit values (otherwise known as internal covariate shift) complicates and slows down the training of deep neural networks. We address this problem by normalizing layer inputs using batch normalization56. Batch normalization allows the use of higher learning rates, reduces the need for careful initialization of training parameters, it acts as a regularizer (sometimes eliminating the need for dropout), and provides faster convergence and higher accuracy rates.

4. Local response normalization - This sort of response normalization implements the concept of lateral inhibition (capacity of an excited neuron to subdue its neighbors) from neurobiology. The output of the nonlinear activation function can result in unbounded activations. Local response normalization (LRN) is used to normalize these activations35. LRN helps to detect high frequency features with a large response and thus it promotes some sort of inhibition. By normalizing around the local neighborhood of a unit/neuron, LRN increases the sensitivity of the neuron compared to its neighbors and thus boosts the neurons with relatively larger activations.

5. Non-linearity - We use the softplus nonlinear activation function57:

softplus(x) = ln(1 + e^x),

where x is the output of each unit at a particular layer. Softplus is smooth and differentiable (near 0) and provided better convergence than the popular rectified linear unit (ReLU), likely because the ReLU hard saturation at 0 can hurt optimization by blocking gradient backpropagation58.

6. Weight initialization - We initialize the weights with random values from a normal distribution with a 0 mean and standard deviation of 0.02.

7. Batch size - We chose a batch size of 128 and applied a mini-batch training strategy in order to reduce loss fluctuation. This batch size facilitated training with our system memory (250GB), and higher-memory systems could benefit from larger batches.

8. Training epochs - We train our network for 8 epochs, terminating training when validation accuracy began to decline.

9. Data shuffling - We introduced data shuffling, applying random orderings for each epoch, in order to break any predefined data structure in the training set.


Diplomska več razlage o teoriji naloge, o FTIR, o metodah, o CNN, o SVM.
Članek pa manj tega in bolj direktno o eksperimentu.