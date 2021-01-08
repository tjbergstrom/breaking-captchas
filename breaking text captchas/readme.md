
This has been done before (there's a reason you don't see these much anymore). Just curious how I would implement it.

1. Data collection. Need lots of labeled captchas. There are/were about 30 different types of text based captchas used by different websites. Start with an easier type, and then move on to more challenging ones.

2. Figure out how to extract characters from a captcha with computer vision tools.

3. Use the extracted characters to train a model.

4. Try it out, test it - trying to break or decode input captchas.

5. Go back and try different CV processing and model hypertunings to improve accuracy.

6. Implement some testing solutions to evaluate performance. For example, a model might be 85% accurate at predicting individual characters, but only 50% accurate at breaking a given captcha.

<br>

![alt text](https://raw.githubusercontent.com/tjbergstrom/Breaking-Captchas/master/breaking%20text%20captchas/assets/screenrecord5.gif)

<br>

7. Start over with different and more challenging types of captchas. To be continued.

<br>
