library(MASS)
library(ggplot2)
library(ggthemes)
library(ggalt)
rm(list=ls(all=TRUE))

################################################################################
# functions
################################################################################
sigmoid <- function(x) {
  y <- 1/(1+exp(-x))
  return(y)
}

logLikelihood <- function(features, labels, weights) {
  scores <- sigmoid(features%*%weights)
  ll <- (t(labels)%*%log(scores) + t(1-labels)%*%log(1-scores))/length(labels)
  return(ll)
}

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  if (is.null(layout)) {
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row, layout.pos.col = matchidx$col))
    }
  }
}


################################################################################
# constants
################################################################################
NUM_ITERATIONS <- 1000
TRAINING_SIZE <- 1024
TEST_SIZE <- 256
LEARNING_RATE <- 0.1/TRAINING_SIZE


################################################################################
# training/test set generation
################################################################################
set.seed(12345)
muA <- c(0, 0)
muB <- c(1, 5)
NUM_FEATURES <- length(muA)
sigma <- matrix(c(1.25, 1.0, 1.0, 1.25), NUM_FEATURES, NUM_FEATURES)

redPoints <- trunc(mvrnorm(TRAINING_SIZE/2, muA, sigma, 0, TRUE)*10^5)/10^5 #truncate to 5th decimal place
bluePoints <- trunc(mvrnorm(TRAINING_SIZE/2, muB, sigma, 0, TRUE)*10^5)/10^5 #truncate to 5th decimal place
trainingSetFeatures <- cbind(matrix(1, TRAINING_SIZE, 1), rbind(redPoints, bluePoints))
trainingSetLabels <- rbind(matrix(0, TRAINING_SIZE/2, 1), matrix(1, TRAINING_SIZE/2, 1))

redPoints <- trunc(mvrnorm(TEST_SIZE/2, muA, sigma, 0, TRUE)*10^5)/10^5 #truncate to 5th decimal place
bluePoints <- trunc(mvrnorm(TEST_SIZE/2, muB, sigma, 0, TRUE)*10^5)/10^5 #truncate to 5th decimal place
testSetFeatures <- cbind(matrix(1, TEST_SIZE, 1), rbind(redPoints, bluePoints))
testSetLabels <- rbind(matrix(0, TEST_SIZE/2, 1), matrix(1, TEST_SIZE/2, 1))


################################################################################
# training
################################################################################
weights <- matrix(0, NUM_FEATURES+1, 1)
likelihood <- matrix(logLikelihood(testSetFeatures, testSetLabels, weights), NUM_ITERATIONS, 1)

for(i in 1:NUM_ITERATIONS) {
  scores <- trainingSetLabels - sigmoid(trainingSetFeatures%*%weights)
  gradients <- t(trainingSetFeatures)%*%scores
  weights <- weights + LEARNING_RATE*gradients
  likelihood[i:NUM_ITERATIONS] <- logLikelihood(testSetFeatures, testSetLabels, weights)
}

testSetFrame = data.frame(cbind(testSetFeatures, testSetLabels))
names(testSetFrame)[NUM_FEATURES+2] <- 'label'

pData <- ggplot(testSetFrame, aes(x=testSetFrame[, 2], y=testSetFrame[, 3], color=as.factor(label)), environment=environment()) +
  geom_point(size=2, show.legend=F) +
  geom_abline(slope = -weights[2]/weights[3], intercept = -weights[1]/weights[3], size=1.5, color="yellow", alpha=0.6) +
  labs(x = "x1", y = "x2") +
  theme_solarized_2(light = FALSE) +
  scale_colour_solarized("red") +
  theme(text = element_text(size=20))
pCurve <-ggplot(data.frame(likelihood), aes(c(1:NUM_ITERATIONS), likelihood), environment=environment()) +
  geom_line(size=1.5, color="yellow", alpha=0.6) +
  labs(x = "iterations", y = "log likelihood") +
  xlim(0, NUM_ITERATIONS) +
  ylim(NA, 0) +
  theme_solarized_2(light = FALSE) +
  scale_colour_solarized("red") +
  theme(text = element_text(size=20))
multiplot(pData, pCurve, cols=1)
