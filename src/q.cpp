
    if(yolo_num%1==0){ 
    // if(yolo_num>0){     
        cv::Mat img1;
        cv::Mat img2;

        auto end11 = std::chrono::high_resolution_clock::now();
        // 设置动态目标的名称列表
        std::vector<string> mvDynamicNames = {
                        "person", "car", 
                        "motorbike", "bus", 
                        "train", "truck", 
                        "boat", "bird", 
                        "cat","dog", 
                        "horse", "sheep", 
                        "crow", "bear"};

        // 初始化 Python 解释器
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('/home/l/slam/cv/YOLO/YOLO8/ultralytics-1113/')"); //显示绝对路径文件夹
        
        pModule = PyImport_ImportModule("c++_python_API-1");  // 替换为你的 Python 模块名

        if (pModule == NULL) {
            std::cerr << "Error: pModule == NULL!!!" << std::endl;
            // return false;
        }
        // std::cout << "YYY 导入   Python 模块和函数 完成" << std::endl;
        
        pFunc = PyObject_GetAttrString(pModule, "process_image_detect_v1");  // 替换为你的 Python 函数名
        
        if (pFunc == NULL) {
            std::cerr << "Error: function process_image not found in the Python module!!!" << std::endl;
            // return false;
        }

        cv::resize(_image, img1, cv::Size(640, 640));// 调整图像大小为 640x640

        // cv::cvtColor(img1, img2, cv::COLOR_BGR2RGB); // 将图像从 BGR 转换为 RGB

        cv::Mat image_detect = img1;

        // cv::cvtColor(img1, image, cv::COLOR_BGR2GRAY); // 将图像从 BGR 转换为 RGB

        // 转换图像数据为 Python 字节对象
        PyObject* pImage = PyBytes_FromStringAndSize((const char*)image_detect.data, image_detect.total());
        if (!pImage) {
            std::cout << "Failed to convert image data to Python bytes object!" << std::endl;
            // Py_DECREF(pModule);
            // // return false;
        }

        // 调用 Python 函数并传递图像数据
        pArgs = PyTuple_New(1);

        PyTuple_SetItem(pArgs, 0, pImage);
        
        auto end12 = std::chrono::high_resolution_clock::now();

        pResult = PyObject_CallObject(pFunc, pArgs);

        auto end13 = std::chrono::high_resolution_clock::now();

        if (!pResult) {
            std::cout << "yolo Failed to call Python function!" << std::endl;
            std::vector<std::vector<int>> matrix(1, std::vector<int>(6, -1)); //其中每一个数都为-1
            // 将矩阵转换为Python对象
            PyObject* pMatrix = PyList_New(matrix.size());
            for (size_t i = 0; i < matrix.size(); i++) {
                PyObject* pRow = PyList_New(matrix[i].size());
                for (size_t j = 0; j < matrix[i].size(); j++) {
                    PyObject* pItem = PyLong_FromLong(matrix[i][j]);
                    PyList_SetItem(pRow, j, pItem);
                }
                PyList_SetItem(pMatrix, i, pRow);
            }

            // 将Python对象赋值给pResult
            pResult = pMatrix;
        }
        
        // 将 Python 返回的对象转换为二维数组
        Py_ssize_t rows, cols; //列   行
        PyObject* pResultIter = PyObject_GetIter(pResult);
        PyObject* pRow = PyIter_Next(pResultIter);
        rows = PyList_Size(pRow);   //列
        cols = PyList_Size(pResult);//行

        for (size_t i=0; i < cols;  ++i)
            {

                PyObject* pRow = PyList_GetItem(pResult, i);

                if (pRow == NULL) {
                    std::cout << "Got NULL item  from pResult at index " << i <<std::endl;
                    continue;
                }

                float x1 = PyLong_AsLong(PyList_GetItem(pRow, 0))*640/640;
                float y1 = PyLong_AsLong(PyList_GetItem(pRow, 1))*480/640;
                float x2 = PyLong_AsLong(PyList_GetItem(pRow, 2))*640/640;
                float y2 = PyLong_AsLong(PyList_GetItem(pRow, 3))*480/640;
                
                int classID = PyLong_AsLong(PyList_GetItem(pRow, 5));//yolo 模式
                float left   = x1;
                float top    = y1;
                float right  = x2;
                float bottom = y2;

                cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
 
                if (classID==0)
                {
                std::vector<float> point_depths; // 新建一个数组用于存储深度值
            
                point_depths.push_back(0);

                std::cout << "@@@"  << std::endl;
                //计算 极限深度 k_d
                for (int k=0;k<_keypoints0.size(); ++k){
                    cv::KeyPoint d_keypoint0 = _keypoints0[k];

                    float kp_u  = d_keypoint0.pt.x;
                    float kp_v = d_keypoint0.pt.y;
                    int box_error =20;
                    if (kp_u > left - 20 && 
                        kp_u < right + 20 && 
                        kp_v > top - 20 && 
                        kp_v < bottom + 20  &&
                        (imDepth.at<float>(kp_v, kp_u) != 0)
                        ) 
                        {
                            float p_depths = imDepth.at<float>(kp_v, kp_u); // 获取深度值
                            p_depths = std::round(p_depths * 10.0f) / 10.0f; // 将深度值精确到小数点后一位
                            point_depths.push_back(p_depths); // 将精确后的深度值存入数组中
                        }
                    }

                std::cout << "@@@"  << std::endl;

                //gmm 1
                //gmm 2

                // 将深度值转换为OpenCV矩阵
                cv::Mat samples(point_depths.size(), 1, CV_32F);
                for (size_t i = 0; i < point_depths.size(); ++i) {
                    samples.at<float>(i, 0) = point_depths[i];
                }

                // 检查样本数量
                int num_samples = samples.rows;
                int num_clusters = 3; // 选择成分数

                double mode_depth_gmm = 0.0;

                if (num_samples >= num_clusters) {
                    // 定义和训练高斯混合模型
                    cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
                    em_model->setClustersNumber(num_clusters);
                    em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
                    em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.1));

                    em_model->trainEM(samples);

                    // 获取每个成分的均值和权重
                    cv::Mat means = em_model->getMeans();
                    cv::Mat weights = em_model->getWeights();

                    // 找到众数（即均值最大的成分）
                    cv::Point max_loc;
                    cv::minMaxLoc(weights, nullptr, nullptr, nullptr, &max_loc);
                    mode_depth_gmm = means.at<double>(max_loc.y, 0);

                    std::cout << "GMM 估计的深度值的众数是: " << mode_depth_gmm << std::endl;
                } else {
                    std::cout << "样本数量不足，无法训练 GMM。"  << mode_depth_gmm  << std::endl;
                    
                }

                float k_d = mode_depth_gmm+1;

                cv::Mat imMask;
                imDepth.copyTo(imMask);

                int mask_area = 0; // 掩码面积

                for (int i = top ; i < bottom; i++) {
                    for (int j = left; j < right; j++) {
                        float depth = imMask.at<float>(i, j);
                        if ( k_d > depth && depth > 0 ) {
                            mask_imResult.at<uchar>(i, j) = 0;
                            mask_area++; // 统计掩码面积
                        }
                    }
                }

                float detection_area = (bottom - top) * (right - left); // 目标检测区域面积

                double rr = 1;
                if (mask_area < rr * detection_area) {
                    // 创建一个矩形区域并将其像素值设置为0
                    cv::Rect detection_rect(left, top, right - left, bottom - top);
                    mask_imResult(detection_rect) = 0;
                }

                }

            }

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10));  // 定义膨胀的结构元素

            //掩玛变小了 这是膨胀操作 膨胀图像中的较亮部分
            cv::erode(mask_imResult, mask_imResult, kernel);  // 对mask_imResult进行腐蚀操作，结果存储在erodedMask中

            Py_DECREF(pResultIter);
            Py_DECREF(pResult);
            // Py_DECREF(pImage);
            Py_DECREF(pArgs);
            Py_DECREF(pFunc);
            Py_DECREF(pModule);


            mask_imResult.copyTo(old_mask_imResult);

            auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(end12 - end11);
            auto duration13 = std::chrono::duration_cast<std::chrono::milliseconds>(end13 - end12);
            std::cout << "Time taken:duration12 " << duration12.count()  <<" "<< duration13.count()<< " milliseconds" << std::endl;
        }else{
            old_mask_imResult.copyTo(mask_imResult); 
            cout<<"n"<<endl;
        }