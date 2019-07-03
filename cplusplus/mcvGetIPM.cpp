#include <iostream>
#include <math.h>
#include <assert.h>
#include <list>
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>



using namespace std;
using namespace cv;

struct IPMInfo
{
    ///min and max x-value on ground in world coordinates
    double xLimits[2];
    ///min and max y-value on ground in world coordinates
    double  yLimits[2];
    ///conversion between mm in world coordinate on the ground
    ///in x-direction and pixel in image
    double  xScale;
    ///conversion between mm in world coordinate on the ground
    ///in y-direction and pixel in image
    double  yScale;
    ///width
    int width;
    ///height
    int height;
    ///portion of image height to add to y-coordinate of
    ///vanishing point
    double  vpPortion;
    ///Left point in original image of region to make IPM for
    double  ipmLeft;
    ///Right point in original image of region to make IPM for
    double  ipmRight;
    ///Top point in original image of region to make IPM for
    double  ipmTop;
    ///Bottom point in original image of region to make IPM for
    double  ipmBottom;
    ///interpolation to use for IPM (0: bilinear, 1:nearest neighbor)
    int ipmInterpolation;
};

struct CameraInfo
{
    ///focal length in x and y
    double  focalLength_x;
    double  focalLength_y;
    ///optical center coordinates in image frame (origin is (0,0) at top left)
    double  opticalCenter_x;
    double  opticalCenter_y;
    ///height of camera above ground
    double  cameraHeight;
    ///pitch angle in radians (+ve downwards)
    double  pitch;
    ///yaw angle in radians (+ve clockwise)
    double  yaw;
    ///width of images
    double  imageWidth;
    ///height of images
    double  imageHeight;
};

struct Point2D
{
    double  x;
    double  y;
};

Point2D mcvGetVanishingPoint(const CameraInfo *cameraInfo)
{
    double  c1 = cos(cameraInfo->pitch*M_PI/180.0);
    double  s1 = sin(cameraInfo->pitch*M_PI/180.0);
    double  c2 = cos(cameraInfo->yaw*M_PI/180.0);
    double  s2 = sin(cameraInfo->yaw*M_PI/180.0);
    //get the vp in world coordinates
    double  vpp[] = {s2/c1,
                                 c2/c1, 0};
    CvMat vp = cvMat(3, 1, CV_64FC1, vpp);
    //transform from world to camera coordinates
    //
    //rotation matrix for yaw
    double  tyawp[] = {c2, -s2, 0,
                                   s2, c2, 0,
                                   0, 0, 1};
    CvMat tyaw = cvMat(3, 3, CV_64FC1, tyawp);
    //rotation matrix for pitch
    double  tpitchp[] = {1, 0, 0,
                                     0, -s1, -c1,
                                     0, c1, -s1};
    CvMat transform = cvMat(3, 3, CV_64FC1, tpitchp);
    //combined transform
    cvMatMul(&transform, &tyaw, &transform);

    //
    //transformation from (xc, yc) in camra coordinates
    // to (u,v) in image frame
    //
    //matrix to shift optical center and focal length
    double  t1p[] = {
            cameraInfo->focalLength_x, 0,
            cameraInfo->opticalCenter_x,
            0, cameraInfo->focalLength_y,
            cameraInfo->opticalCenter_y,
            0, 0, 1};
    CvMat t1 = cvMat(3, 3, CV_64FC1, t1p);
    //combine transform
    cvMatMul(&t1, &transform, &transform);
    //transform
    cvMatMul(&transform, &vp, &vp);

    //
    //clean and return
    //
    Point2D ret;
    ret.x = cvGetReal1D(&vp, 0);
    ret.y = cvGetReal1D(&vp, 1);
    return ret;
}


void mcvTransformImage2Ground( Mat *inPoints, Mat *outPoints, const CameraInfo *cameraInfo)
{

    //add two rows to the input points
    Mat inPoints4 = *inPoints;
    cv::Mat row_1 = cv::Mat::ones(1, inPoints4.cols, CV_64FC1);
    cv::Mat row_2 = cv::Mat::zeros(1, inPoints4.cols,CV_64FC1);
    inPoints4.push_back(row_1);
    inPoints4.push_back(row_2);


    Mat inPoints2, inPoints3, inPointsr4, inPointsr3;
    inPoints2 = inPoints4.rowRange(0,2);
    inPoints3 = inPoints4.rowRange(0,3);
    inPointsr3 = inPoints4.rowRange(2,3);
    inPointsr4 = inPoints4.rowRange(3,4);


    //create the transformation matrix
    double c1 = cos(cameraInfo->pitch*M_PI/180.0);
    double s1 = sin(cameraInfo->pitch*M_PI/180.0);
    double c2 = cos(cameraInfo->yaw*M_PI/180.0);
    double s2 = sin(cameraInfo->yaw*M_PI/180.0);

    double  matp[] = {
            -cameraInfo->cameraHeight*c2/cameraInfo->focalLength_x,
            cameraInfo->cameraHeight*s1*s2/cameraInfo->focalLength_y,
            (cameraInfo->cameraHeight*c2*cameraInfo->opticalCenter_x/
             cameraInfo->focalLength_x)-
            (cameraInfo->cameraHeight *s1*s2* cameraInfo->opticalCenter_y/
             cameraInfo->focalLength_y) - cameraInfo->cameraHeight *c1*s2,

            cameraInfo->cameraHeight *s2 /cameraInfo->focalLength_x,
            cameraInfo->cameraHeight *s1*c2 /cameraInfo->focalLength_y,
            (-cameraInfo->cameraHeight *s2* cameraInfo->opticalCenter_x
             /cameraInfo->focalLength_x)-(cameraInfo->cameraHeight *s1*c2*
                                          cameraInfo->opticalCenter_y /cameraInfo->focalLength_y) -
            cameraInfo->cameraHeight *c1*c2,

            0,
            cameraInfo->cameraHeight *c1 /cameraInfo->focalLength_y,
            (-cameraInfo->cameraHeight *c1* cameraInfo->opticalCenter_y /
             cameraInfo->focalLength_y) + cameraInfo->cameraHeight *s1,

            0,
            -c1 /cameraInfo->focalLength_y,
            (c1* cameraInfo->opticalCenter_y /cameraInfo->focalLength_y) - s1,
    };
    Mat mat = Mat(4, 3, CV_64FC1, matp);
    inPoints4 = mat*inPoints3;
    inPointsr4 = inPoints4.rowRange(3,4);
    inPoints4.row(0)=inPoints4.row(0)/inPointsr4;
    inPoints4.row(1)=inPoints4.row(1)/inPointsr4;
    Mat inPoints5;
    inPoints5 = inPoints4.rowRange(0,2);
    *outPoints = inPoints5;


}


void mcvTransformGround2Image(Mat *inPoints,
                              Mat *outPoints, const CameraInfo *cameraInfo) {
    //add two rows to the input points
    Mat inPoints3 = *inPoints;
    cv::Mat row_1 = cv::Mat::ones(1, inPoints3.cols, CV_64FC1) *(- cameraInfo->cameraHeight);
    inPoints3.push_back(row_1);
    //copy inPoints to first two rows
    Mat inPoints2, inPointsr3;
    inPoints2 = inPoints3.rowRange(0, 2);
    inPointsr3 = inPoints3.rowRange(2, 3);

    //create the transformation matrix
    double  c1 = cos(cameraInfo->pitch*M_PI/180);
    double  s1 = sin(cameraInfo->pitch*M_PI/180);
    double  c2 = cos(cameraInfo->yaw*M_PI/180);
    double  s2 = sin(cameraInfo->yaw*M_PI/180);
    double  matp[] = {
            cameraInfo->focalLength_x * c2 + c1*s2* cameraInfo->opticalCenter_x,
            -cameraInfo->focalLength_x * s2 + c1*c2* cameraInfo->opticalCenter_x,
            - s1 * cameraInfo->opticalCenter_x,

            s2 * (-cameraInfo->focalLength_y * s1 + c1* cameraInfo->opticalCenter_y),
            c2 * (-cameraInfo->focalLength_y * s1 + c1* cameraInfo->opticalCenter_y),
            -cameraInfo->focalLength_y * c1 - s1* cameraInfo->opticalCenter_y,

            c1*s2,c1*c2,-s1
    };

    Mat mat = Mat(3, 3, CV_64FC1, matp);
    //multiply
    cout<<"inPoints3 x:"<<inPoints3.at<double>(0,0)<<"y:"<<inPoints3.at<double>(1,0)<<endl;
    cout<<"inPoints3 x:"<<inPoints3.at<double>(0,1)<<"y:"<<inPoints3.at<double>(1,1)<<endl;
    inPoints3 = mat * inPoints3;
    //divide by last row of inPoints3
    inPointsr3 = inPoints3.row(2);
    //inPoints3.row(0) = inPoints3.row(0) / inPointsr3;
    //inPoints3.row(1) = inPoints3.row(1) / inPointsr3;
    for (int ii = 0; ii < inPoints3.cols; ++ii ) {
        inPoints3.at<double>(0,ii) = inPoints3.at<double>(0,ii) / inPointsr3.at<double>(0,ii);
        inPoints3.at<double>(1,ii) = inPoints3.at<double>(1,ii) / inPointsr3.at<double>(0,ii);
    }
    Mat inPoints5;
    inPoints5 = inPoints3.rowRange(0,2);
    cout<<"inPoints5 x:"<<inPoints5.at<double>(0,0)<<"y:"<<inPoints5.at<double>(1,0)<<endl;
    cout<<"inPoints5 x:"<<inPoints5.at<double>(0,1)<<"y:"<<inPoints5.at<double>(1,1)<<endl;
    *outPoints = inPoints5;

}


int main()
{
    Mat* inImage;
    Mat* outImage;
    IPMInfo *ipmInfo;
    CameraInfo *cameraInfo;

    //im = cvLoadImage("/home/loc/CLionProjects/untitled1/01.jpg", CV_RGB2BGRA);
    //cvSaveImage( "/home/loc/CLionProjects/untitled1/02.jpg", im );
    // convert to mat and get first channel
    Mat im_mat = imread("/home/loc/CLionProjects/untitled1/00195.jpg", CV_RGB2BGRA);
    inImage =&im_mat;
    //cout<<"im_mat:"<<(float)im_mat.at<uchar>(0,0)<<endl;
    namedWindow("显示窗口");
    imshow("输入图片",im_mat);


    double vpPortion = 0.06;
    //将CameraInfo_1的地址赋予CameraInfo
    CameraInfo CameraInfo_1;
    cameraInfo = &CameraInfo_1;
    cameraInfo->focalLength_x = 1456.43;
    cameraInfo->focalLength_y = 1465.53;
    cameraInfo->opticalCenter_x = 605.35;
    cameraInfo->opticalCenter_y = 292.04;
    cameraInfo->cameraHeight = 1310.0;
    cameraInfo->pitch = -7.25;
    cameraInfo->yaw = 0.0;
    cameraInfo->imageWidth = 1280.0;
    cameraInfo->imageHeight = 720.0;

    //get size of input image
    double  u, v;
    v = inImage->rows;
    u = inImage->cols;

    Mat om_mat = Mat(v,u,CV_64FC1);
    outImage = &om_mat;

    //get the vanishing point
    Point2D vp;
    vp = mcvGetVanishingPoint(cameraInfo);
    vp.y = MAX(0, vp.y);

    //get extent of the image in the xfyf plane
    double  eps = vpPortion * v; //VP_PORTION*v;

    IPMInfo ipmInfo_1;
    ipmInfo = &ipmInfo_1;
    ipmInfo_1.ipmLeft = 0;
    ipmInfo_1.ipmRight= u -1;
    ipmInfo_1.ipmTop = vp.y+eps;
    ipmInfo_1.ipmBottom = v-1;

    double  uvLimitsp[] = {vp.x,ipmInfo->ipmRight, ipmInfo->ipmLeft, vp.x,ipmInfo->ipmTop, ipmInfo->ipmTop,   ipmInfo->ipmTop,  ipmInfo->ipmBottom};
    //{vp.x, u, 0, vp.x,
    //vp.y+eps, vp.y+eps, vp.y+eps, v};
    Mat uvLimits = Mat(2, 4, CV_64FC1, uvLimitsp);
    cout<<"uvLimits:"<<uvLimits<<endl;

    //get these points on the ground plane
    Mat xyLimitsp = Mat(2, 4, CV_64FC1);
    Mat xyLimits = xyLimitsp;
    mcvTransformImage2Ground(&uvLimits, &xyLimits,cameraInfo);

    Mat row1, row2;
    row1 = xyLimits.row(0);
    row2 = xyLimits.row(1);
    double xfMax, xfMin, yfMax, yfMin;
    cv::minMaxIdx(row1,&xfMin,&xfMax);
    cv::minMaxIdx(row2,&yfMin,&yfMax);


    int outRow = outImage->rows;
    int outCol = outImage->cols;

    double  stepRow = (yfMax-yfMin)/outRow;
    double  stepCol = (xfMax-xfMin)/outCol;

    //construct the grid to sample
    Mat xyGrid = Mat(2, outRow*outCol, CV_64FC1);
    int i, j;
    double  x, y;
    for (i=0, y=yfMax-0.5*stepRow; i<outRow; i++, y-=stepRow)
        for (j=0, x=xfMin+0.5*stepCol; j<outCol; j++, x+=stepCol)
        {
            xyGrid.at<double >(0 , i*outCol+j) = x ;
            xyGrid.at<double >(1 , i*outCol+j) = y ;
        }

    //get their pixel values in image frame
    Mat uvGrid = cvCreateMat(2, outRow*outCol, CV_64FC1);
    mcvTransformGround2Image(&xyGrid, &uvGrid, cameraInfo);


    double  means_1 = 0;
    for(int i = 0;i<v;i++)
        for(int j =0;j<u;j++)
        {
            means_1 = means_1 + (double )im_mat.at<uchar>(i,j);
        }
    double  means_2 = means_1/(v*u);
    double  means = means_2/255;

    int ui,vi;
    for (i=0; i<outRow; i++)
        for (j=0; j<outCol; j++) {

            /*get pixel coordiantes*/ \
           ui = (double ) uvGrid.at<double >(0, i * outCol + j);

           vi = (double ) uvGrid.at<double >(1, i * outCol + j);
            //cout<<"ui:"<<ui<<"vi:"<<vi<<endl;

            if (ui < ipmInfo->ipmLeft || ui > ipmInfo->ipmRight || vi < ipmInfo->ipmTop || vi > ipmInfo->ipmBottom) {

                om_mat.at<double >(i, j) = means;

            }
                /*not out of bounds, then get nearest neighbor*/ \
       else {
                /*Bilinear interpolation*/

                int x1 = int(ui), x2 = int(ui + 1);
                int y1 = int(vi), y2 = int(vi + 1);
                double  x = ui - x1, y = vi - y1;
                double  val = (double ) im_mat.at<uchar>(y1, x1) * (1 - x) * (1 - y) +
                            (double ) im_mat.at<uchar>(y1, x2) * (1 - x) * (1 - y) +
                            (double ) im_mat.at<uchar>(y2, x1) * (1 - x) * y +
                            (double ) im_mat.at<uchar>(y2, x2) * x * y;

                om_mat.at< double>(i, j) = (double)val/255;
                cout<<"VAL:"<<val<<endl;
            }


        }
    imshow("dasd",om_mat);
    cout<<"om_mat:"<<om_mat<<endl;
    waitKey(0);
    return 0;
}