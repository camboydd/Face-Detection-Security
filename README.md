# Face Detection Security
 
### About
> **_NOTE:_**  This project uses dlib (models folder) to detect and recognize faces.

This project takes streaming video, detects and recognizes faces in the video. It will recognize the people (faces) that are uploaded into the correct folder (instructions below). It will blur all other faces that are detected but not recognized.

A script like this could be useful for someone recording content in public, and want to automatically blur all of the faces besides those which are defined.

### Examples
<img width="600" alt="Screenshot 2024-02-15 at 4 04 22 PM" src="https://github.com/camboydd/Computer-Vision-Project/assets/98326906/6061cc00-5994-4d4b-a9ad-781e4f1bd367">

<img width="600" alt="Screenshot 2024-02-15 at 4 03 55 PM" src="https://github.com/camboydd/Computer-Vision-Project/assets/98326906/bc01726d-7c78-459e-8676-b278ed2df6ec">

### How to run
- Clone repository to your machine
- Download python3 and all required packages
- Create a folder in the root directory, called "faces"
- Create a folder titled <your_name> in "faces"
- Insert images (.png, .jpg, .jpeg) of your face into <your_name>
- Run python3 face_rec.py in your root directory terminal
