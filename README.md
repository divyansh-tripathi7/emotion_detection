# Emotion Detection from Scratch

**Description:** This project aims to train an image classifier from scratch to detect emotions from facial expressions using the Kaggle FER-2013 Dataset.

## Approaches

We employed the following approaches:

1. **Custom CNN From Scratch**:
   - A Convolutional Neural Network (CNN) was built from scratch, using layers like Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, and Dense.
   - The model architecture was designed to gradually extract features from the input images and classify them into seven emotion categories.
   - We trained this on FER dataset

2. **Custom CNN With Augmentation**:
   - To enhance the model's performance and generalization ability, data augmentation techniques were applied during training.
   - This included random rotations, shifts, shears, zooms, and horizontal flips of the training images.
   - The augmented dataset helped the model learn to recognize emotions from various perspectives and variations, improving robustness.
   - We trained this on FER dataset

3. **Transfer Learning with VGG16**:
   - We leveraged the pre-trained VGG16 model, a powerful CNN architecture, for transfer learning.
   - The VGG16 model was fine-tuned on the emotion detection task by replacing the final classification layer and training it on the FER-2013 dataset.
   - Transfer learning allowed us to benefit from the knowledge learned by VGG16 on a large image dataset, speeding up training and potentially improving performance.
   - We trained this on FER dataset

4. **Transfer Learning with ResNet50**:
   - Similar to VGG16, we also explored transfer learning with the ResNet50 model, another widely used CNN architecture.
   - The ResNet50 model was fine-tuned on the FER-2013 dataset, adapting its learned features to the emotion detection task.
   - We compared the results of transfer learning with VGG16 and ResNet50 to identify the most effective approach.
   - We trained this on FER dataset

## Things We Did

- **Data Cleaning**: Before training, we cleaned the dataset by removing any corrupted or irrelevant files. This ensured the quality and consistency of the training data.
- **Data Analysis**: We performed data analysis to understand the distribution of emotions in the dataset. This helped us gain insights into potential challenges and guide model development.
- **Model Evaluation**: After training, we evaluated the performance of our models using metrics like accuracy, precision, recall, and F1-score. We also visualized the results with confusion matrices and classification reports.
- **Hyperparameter Tuning**: We experimented with different hyperparameters, such as learning rate, batch size, and dropout rate, to optimize the models' performance.

## Results

Our experiments showed that transfer learning with VGG16 and ResNet50 achieved significantly better results compared to the custom CNN from scratch. Image augmentation further enhanced the models' performance, leading to improved accuracy on the test set.

## Future Work

- We could explore other transfer learning models or custom CNN architectures to further improve the accuracy and efficiency of emotion detection.
- We could incorporate additional data sources or techniques, such as facial landmark detection, to enhance the model's understanding of facial expressions.

## Conclusion

This project demonstrated the effectiveness of using transfer learning and data augmentation to build an accurate emotion detection model. Our findings contribute to the ongoing research in computer vision and human-computer interaction, with potential applications in fields like healthcare, education, and entertainment.
