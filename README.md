# numbio: Numerical Biomarkers Design Algorithm

### Welcome to our numbio project!

<p align="center"> 
<img src="./images/cost_function.png" title="Which one would you rather
minimize?" alt="Cost function">
</p>

### What is this?

This the code (written in Python) used to compute numerical biomarkers.
Numerical biomarkers are features designed to easily solve inverse and
classification problems.
For more details about the method and its applications, we refer you to our poster
(poster.pdf) and to our publication (coming soon).

### Requirements

This code works with both versions 2.7 and 3.4 of Python and does not require
any external package.
For the production of figures (optional), both versions 1.1.1 and 1.5.1 of matplotlib work fine.

### Try the demo!

There is no better way to understand the method than to try it yourself!
Follow the instructions to compute numerical biomarkers for a very simple test
case.

> python numericalBiomarkers.py demo 0

The command to be executed in general is the following:

> python numericalBiomarkers.py your_case_name plotFigures

where plotFigures is 0 (no figure) or 1 (plot figures).

### How it works

#### Data

The data required by the algorithm to compute the numerical biomarkers consists of parameter samples and associated model outputs: a file 'params.txt' (space seperated columns, ASCII format) of size [numRows, numCols] = [*N* X *d*] where *N* is the number of samples and *d* the number of parameters and a file 'outputs.txt' (same format) of size [*N* X *ny*] where *ny* is the number of dictionary entries. Both these files must be in you case directory (e.g. ./data/demo).
For memory and speed improvements, you can use a binary format for these files. See our section "Binary format" below.

#### Hyper-parameters

Some hyper-parameters of the method need to be specified in a 'numbio.in' file in your case directory. The file should have the following format:


  * first row: numSamples minIter maxIter tolStagnation stepSize
  * second row: lambda_1 lambda_2 ... lambda_d

Let's explain the hyper-parameters:

      * numSamples: number of samples to be considered (must be <= N).
      * minIter: minimum number of iterations carried out during the Nesterov gradient descent.
      * maxIter: maximum number of these iterations.
      * tolStagnation: stagnation tolerance on the cost function. Both maxIter and tolStagnation are stopping criteria.
      * stepSize: initial step size of the Nesterov gradient descent.
      * lambda_k: l1 penalization hyper-parameter for the kth parameter.

#### Output

After executing the program, a file "weights_[your_case_name].txt" should be written in the current directory. It is of size [*d*+2, *ny*]. The first *d* rows are the dictionary entries weights, the last two rows are respectively vectors *nu* and *tau* used to rescale the dictionary entries.


#### Binary format

If your ASCII files have the proper format (see "Data" section above), you can convert them to our binary format. Simply run:
> python ./utils/compress.py
[your_file_in_ascii_format_with_txt_extension]

The new files should have the same name but with a ".bin" extension.
The code in numericalBiomarkers.py will automatically select the binary format if they are stored in the same directory as the ASCII files (./data/your_case/).

Conversely, if you want to convert binary files to
ascii files, execute the following command:
> python ./utils/deflate.py [file_in_binary_format_with_bin_extension]