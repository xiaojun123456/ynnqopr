//
// Created by 庾金科 on 23/10/2017.
//

#include "../include/Pipeline.h"


namespace pr {

    const int HorizontalPadding = 4;

    std::vector<std::string> chars_code{"京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

    PipelinePR::PipelinePR(std::string detector_filename,
                           std::string finemapping_prototxt,
                           std::string finemapping_caffemodel,
                           std::string segmentation_prototxt,
                           std::string segmentation_caffemodel,
                           std::string charRecognization_proto,
                           std::string charRecognization_caffemodel,
                           std::string segmentationfree_proto,
                           std::string segmentationfree_caffemodel) {

        plateDetection = new PlateDetection(detector_filename);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);
        plateSegmentation = new PlateSegmentation(segmentation_prototxt, segmentation_caffemodel);
        generalRecognizer = new CNNRecognizer(charRecognization_proto, charRecognization_caffemodel);

        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(segmentationfree_proto, segmentationfree_caffemodel);

    }

    PipelinePR::~PipelinePR() {

        delete plateDetection;
        delete fineMapping;
        delete plateSegmentation;
        delete generalRecognizer;

        delete segmentationFreeRecognizer;

    }

    std::vector<PlateInfo> PipelinePR:: RunPiplineAsImage(cv::Mat plateImage) {

        std::vector<PlateInfo> results;
        std::vector<pr::PlateInfo> plates;

        plateDetection->plateDetectionRough(plateImage,plates);

        for (pr::PlateInfo plateinfo:plates) {

            cv::Mat image_finemapping = plateinfo.getPlateImage();

            image_finemapping = fineMapping->FineMappingVertical(image_finemapping);

            image_finemapping = pr::fastdeskew(image_finemapping, 5);

            image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 2, 5);

            cv::resize(image_finemapping, image_finemapping, cv::Size(136, 36));
            plateinfo.setPlateImage(image_finemapping);
            std::vector<cv::Rect> rects;
            plateSegmentation->segmentPlatePipline(plateinfo, 1, rects);
            plateSegmentation->ExtractRegions(plateinfo, rects);

            cv::copyMakeBorder(image_finemapping, image_finemapping, 0, 0, 0, 20, cv::BORDER_REPLICATE);

            plateinfo.setPlateImage(image_finemapping);
            generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
            plateinfo.decodePlateNormal(chars_code);
            results.push_back(plateinfo);
            std::cout << plateinfo.getPlateName() << std::endl;


        }

        return results;

    }

    std::vector<cv::Point> PipelinePR:: GetPointFromPlateRough(cv::Mat plateImage) {

        std::vector<cv::Point> results;
        std::vector<pr::PlateInfo> plates;

        plateDetection->plateDetectionRough(plateImage,plates,36,700);

        for (pr::PlateInfo plateinfo:plates) {

            //获得目标区域矩形
            cv::Rect r = plateinfo.getPlateRect();

            //获得车牌的4个拐点
            std::vector<cv::Point> cors = fineMapping->get4point(plateinfo.getPlateImage());

            //目标区域的坐标转换到原始图片中
            //left top
            cors[0].x += r.x;
            cors[0].y += r.y;

            //right top
            cors[1].x += r.x;
            cors[1].y += r.y;

            //right bottom
            cors[2].x += r.x;
            cors[2].y += r.y;

            //left bottom
            cors[3].x += r.x;
            cors[3].y += r.y;


            //粗略的轮廓拐点
//            cors[0].x = r.x;
//            cors[0].y = r.y;
//            cors[1].x = r.x+r.width;
//            cors[1].y = r.y;
//            cors[2].x = r.x+r.width;
//            cors[2].y = r.y+r.height;
//            cors[3].x = r.x;
//            cors[3].y = r.y+r.height;

            results = cors;

        }

        return results;

    }

    std::pair<std::string,std::vector<cv::Point>> PipelinePR:: GetPointFromPlateRough2(cv::Mat plateImage) {

        std::vector<cv::Point> results;
        std::vector<pr::PlateInfo> plates;

        std::string str="";

        plateDetection->plateDetectionRough(plateImage,plates,36,700);

        for (pr::PlateInfo plateinfo:plates) {

            //获得目标区域矩形
            cv::Rect r = plateinfo.getPlateRect();

            std::vector<cv::Rect> rects(7);
            rects = fineMapping->get7Rect(plateinfo.getPlateImage());

            std::vector<cv::Point> cors(4);

            if(rects[0].area()==0){
                //粗略的轮廓拐点
                cors[0].x = r.x;
                cors[0].y = r.y;
                cors[1].x = r.x+r.width;
                cors[1].y = r.y;
                cors[2].x = r.x+r.width;
                cors[2].y = r.y+r.height;
                cors[3].x = r.x;
                cors[3].y = r.y+r.height;
                results = cors;

                break;

            }else{

                cors[0].x=rects[0].x;
                cors[0].y=rects[0].y;

                cors[1].x=rects[6].x+rects[6].width;
                cors[1].y=rects[6].y;

                cors[2].x=rects[6].x+rects[6].width;
                cors[2].y=rects[6].y+rects[6].height;

                cors[3].x=rects[0].x;
                cors[3].y=rects[0].y+rects[0].height;


                //目标区域的坐标转换到原始图片中
                //left top
                cors[0].x += r.x;
                cors[0].y += r.y;

                //right top
                cors[1].x += r.x;
                cors[1].y += r.y;

                //right bottom
                cors[2].x += r.x;
                cors[2].y += r.y;

                //left bottom
                cors[3].x += r.x;
                cors[3].y += r.y;

                results = cors;

                plateSegmentation->ExtractRegions(plateinfo, rects);
                generalRecognizer->SegmentBasedSequenceRecognition(plateinfo);
                str = plateinfo.decodePlateNormal(pr::CH_PLATE_CODE);


                break;
            }

        }

        std::pair<std::string,std::vector<cv::Point>> pair;
        pair.first = str;
        pair.second = results;

        return pair;

    }

    std::pair<std::string,float> PipelinePR::GetPlateResult(cv::Mat plateImage,
                                                            std::vector<cv::Point> cors){

        std::vector<cv::Point2f> cors2(4);
        cors2[0] = cv::Point2f(cors[2].x, cors[2].y);
        cors2[1] = cv::Point2f(cors[3].x, cors[3].y);
        cors2[2] = cv::Point2f(cors[1].x, cors[1].y);
        cors2[3] = cv::Point2f(cors[0].x, cors[0].y);

        std::vector<cv::Point2f> corners_trans2(4);
        corners_trans2[0] = cv::Point2f(136, 36);
        corners_trans2[1] = cv::Point2f(0, 36);
        corners_trans2[2] = cv::Point2f(136, 0);
        corners_trans2[3] = cv::Point2f(0, 0);

        cv::Mat transform = cv::getPerspectiveTransform(cors2, corners_trans2);
        cv::Mat quad = cv::Mat::zeros(36, 136, CV_8UC3);
        cv::warpPerspective(plateImage, quad, transform, quad.size());

        cv::resize(quad, quad, cv::Size(136+4, 36));

        std::pair<std::string,float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(quad,pr::CH_PLATE_CODE);

        return res;

    };


}