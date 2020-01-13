## CNN with Different Initialization
I am currently learning about padding in CNNs through **[Sebastian Raschka's](http://pages.stat.wisc.edu/~sraschka/teaching/)** Deep Learning Material.
<br>
<br> 
Padding is when *we add an additional layer to the image in order to avoid shrinking outputs and/or loosing information on the corners of an image* when dealing with Convolutional Neural Networks.
<br>
<br>
To get a better grasp of this, look at the attached image in this repository, where we see how output size, padding pixels, kernel size, and stride all contribute to our output.
<br>
<br>
The following script goes through this process: convolving and padding a number of times, changing the dimensions of the image without loosing important information.
<br>
<br>
 
