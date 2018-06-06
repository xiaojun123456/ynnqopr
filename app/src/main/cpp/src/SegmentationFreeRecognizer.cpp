#include "../include/SegmentationFreeRecognizer.h"

namespace pr {

    SegmentationFreeRecognizer::SegmentationFreeRecognizer(std::string prototxt,
                                                           std::string caffemodel) {
        //从磁盘直接加载序列化模型
        net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    }

    //内联函数
    inline int judgeCharRange(int id){
        return id<31 || id>63;
    }

    std::pair<std::string,float> decodeResults(cv::Mat code_table,
                                               std::vector<std::string> mapping_table,
                                               float thres){

        cv::MatSize mtsize = code_table.size;
        int sequencelength = mtsize[2];
        int labellength = mtsize[1];

        //std::cout <<"mtsize:"<< labellength << "--" << sequencelength << std::endl;

        cv::transpose(code_table.reshape(1,1).reshape(1,labellength),code_table);

        std::string name = "";

        std::vector<int> seq(sequencelength);

        std::vector<std::pair<int,float>> seq_decode_res;

        for(int i = 0 ; i < sequencelength;  i++) {
            float *fstart = ((float *) (code_table.data) + i * labellength );
            int id = std::max_element(fstart,fstart+labellength) - fstart;
            seq[i] =id;
        }

        float sum_confidence = 0;

        int plate_lenghth  = 0 ;

        for(int i = 0 ; i< sequencelength ; i++)
        {
            if(seq[i]!=labellength-1 && (i==0 || seq[i]!=seq[i-1]))
            {
                float *fstart = ((float *) (code_table.data) + i * labellength );
                float confidence = *(fstart+seq[i]);
                std::pair<int,float> pair_(seq[i],confidence);
                seq_decode_res.push_back(pair_);
            }
        }

        int  i = 0;

        if(judgeCharRange(seq_decode_res[0].first) && judgeCharRange(seq_decode_res[1].first))
        {
            i=2;
            int c = seq_decode_res[0].second<seq_decode_res[1].second;
            name+=mapping_table[seq_decode_res[c].first];
            sum_confidence+=seq_decode_res[c].second;
            plate_lenghth++;
        }

        for(; i < seq_decode_res.size();i++)
        {
            name+=mapping_table[seq_decode_res[i].first];
            sum_confidence +=seq_decode_res[i].second;
            plate_lenghth++;
        }

        std::pair<std::string,float> res;
        res.second = sum_confidence/plate_lenghth;
        res.first = name;

        return res;

    }

    std::string decodeResults(cv::Mat code_table,
                              std::vector<std::string> mapping_table){

        cv::MatSize mtsize = code_table.size;
        int sequencelength = mtsize[2];//16
        int labellength = mtsize[1];//84

        std::cout <<"mtsize sequencelength--labellength:"<< sequencelength << "--" << labellength <<"-chars-"<<mapping_table.size()<<"--"<<mapping_table[mapping_table.size()-1]<< std::endl;

        cv::transpose(code_table.reshape(1,1).reshape(1,labellength),code_table);

        std::string name = "";
        std::vector<int> seq(sequencelength);
        for(int i = 0 ; i < sequencelength;  i++) {
            float *fstart = ((float *) (code_table.data) + i * labellength );
            int id = std::max_element(fstart,fstart+labellength) - fstart;
            seq[i] =id;
            std::cout << "i--id :" << i << "--" << id << std::endl;
        }
        std::cout << "---------------" << std::endl;
        for(int i = 0 ; i< sequencelength ; i++)
        {
            if (seq[i] != labellength - 1 && (i == 0 || seq[i] != seq[i - 1])) {
                name += mapping_table[seq[i]];
                std::cout <<"i--id :"<< i << "--" << seq[i] <<"--"<< mapping_table[seq[i]] << std::endl;
            }
        }

        std::cout << name << std::endl;

        std::cout << "---------------" << std::endl;

        return name;
    }

    std::pair<std::string,float> SegmentationFreeRecognizer::SegmentationFreeForSinglePlate(cv::Mat Image,
                                                                                            std::vector<std::string> mapping_table) {

        //std::cout << Image << std::endl;//140x36

        //图片先顺时针旋转90度再翻转
        cv::transpose(Image,Image);
        //cv::imwrite("../lpr/general_test/cache/6-transpose.png", Image);

        //std::cout << Image << std::endl;

        //创建4维二进制大对象
        cv::Mat inputBlob = cv::dnn::blobFromImage(Image, 1 / 255.0, cv::Size(40,160));

        //set the network input, "data" is the name of the input layer
        net.setInput(inputBlob, "data");

        //向前传播我们的图像，获取分类结果。//compute output, "prob" is the name of the output layer
        //cv::Mat char_prob_mat = net.forward();
        cv::Mat char_prob_mat = net.forward("prob");

        //std::cout << char_prob_mat << std::endl;

        //decodeResults(char_prob_mat, mapping_table);

        return decodeResults(char_prob_mat,mapping_table,0.00);

    }


}