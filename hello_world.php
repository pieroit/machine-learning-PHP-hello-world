<?php

include "vendor/autoload.php";

use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WordTokenizer;
use Phpml\CrossValidation\RandomSplit;
use Phpml\Classification\DecisionTree;
use Phpml\Classification\NaiveBayes;
use Phpml\Metric\Accuracy;
use Phpml\Metric\ConfusionMatrix;

srand(1);

$texts     = [];
$sentiment = [];
$row_index     = 0;
$max_row_index = 2000;

$handle = fopen("data/Womens Clothing E-Commerce Reviews.csv", "r");
while ( ($row = fgetcsv($handle, 0, ',')) !== FALSE && ($row_index<$max_row_index) ) {
    if($row[6]==0 || rand(0,3)==0) {
        $texts[]     = $row[4];
        $sentiment[] = $row[6];
        $row_index++;
    }
}

$vectorizer = new TokenCountVectorizer(new WordTokenizer());
$vectorizer->fit($texts);
$vectorizer->transform($texts);

$dataset = new ArrayDataset($texts, $sentiment);
$split_dataset = new RandomSplit($dataset);
$X_train = $split_dataset->getTrainSamples();
$y_train = $split_dataset->getTrainLabels();
$X_test  = $split_dataset->getTestSamples();
$y_test  = $split_dataset->getTestLabels();

$model = new NaiveBayes();
$model->train( $X_train, $y_train );

$p_train = $model->predict( $X_train );
$accuracy_train = Accuracy::score( $y_train, $p_train );
echo("\n\nTrain accuracy: " . $accuracy_train);

$p_test = $model->predict( $X_test );
$accuracy_test   = Accuracy::score( $y_test, $p_test );
echo("\n\nTest accuracy: " . $accuracy_test);

$confusion_matrix = ConfusionMatrix::compute($y_test, $p_test);
echo("\n\nConfusion Matrix\n");
print_r($confusion_matrix);



$new = [
    "I look fat, it's expensive and too large.",
    "This is a lovely t-shirt, colorful and comfy.",
    "Not that bad, but I'm not recommending it.",
    "This is soooo nice! Took it in two different colors."
];
$vectorizer->transform($new);
$prediction = $model->predict($new);
print_r($prediction);

