to do:
add TR start times to preprocessing
scanner (via forp--> usb) outputs "5" keypress at start of each scan
- intercept hid, filter "5", send message that image was acquired
- pass along other keypresses

## decoding
- build decoder for alex's semantics paper
- read previous papers

# applications

## wow
- soundtrack generator
  - label movies (/other stimuli) with basic emotions
  - train encoding model on subject responses to labeled movies
  - run subject on novel movies and decode basic emotion
  - trigger sound samples in response

## decoding
- decoder for Alex narrative semantics data
  - encoding model is already built
  - decoding model

## science
- [bayesian experimental design](https://en.wikipedia.org/wiki/Bayesian_experimental_design)

# detrending
## validation
- correlation with offline preprocessed
  - get offline preprocessed data
- explained variance

- understand state of art detrending
	- in lab
	- literature review
- determine candidates and evaluation metrics
- acquire suitable testing data
- write evaluation code
- run tests