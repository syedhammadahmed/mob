# MOB Dataset

The dataset frames can be found [here](https://drive.google.com/file/d/1Zjib-WaF5hk3wVrj5eW6ewdpMFcn45Wo/view?usp=drive_link).

The dataset annotation labels can be found [here](https://github.com/syedhammadahmed/mob/blob/main/mob.csv).

## **Dataset Attributes**

**videoID:** YouTube video id

**startingTimeStamp:**  starting timestamp (seconds) of clip relative to the original/published YouTube video

**endingTimeStamp:**  ending timestamp (seconds) of clip relative to the original/published YouTube video 

**is_anime:** the video is an anime or not

**is_videoGame:** the video is a gameplay of a video-game or not

**is_fastRepetitiveMovement:** the video contains fast and repetitive motions or not

**is_cartoonCharacter:** the video contains a cartoon character or not

**is_appearanceUnpleasant:** the appearance of the cartoon character is unpleasant/disgusting or not

**is_violenceActivity:** the video containts any violent activity (hitting/destruction/killing) or not

**is_obscene:** the video containts any obscene/indecent activity or not

**is_audio:** the video containts any audio

**is_loud:** the video containts any loud music/noise or not

**is_screaming:** the video containts any screaming/shouting or not

**is_explosion:** the video containts any explosion or gunshot sounds or not

**videoClass:** Ground Truth Label based on video features - Malicious (1) or Benign (0)

**audioClass:** Ground Truth Label based on audio features - Malicious (1) or Benign (0)



## **Citation**

If you find our work useful in your research, please cite:
```
 @inproceedings{ahmed2023malicious,
  title={Malicious or Benign? Towards Effective Content Moderation for Children's Videos},
  author={Ahmed, Syed Hammad and Khan, Muhammad Junaid and Qaisar, Hafiz Muhammad Umer and Sukthankar, Gita},
  booktitle={The International FLAIRS Conference Proceedings},
  volume={36},
  year={2023}
}
```
