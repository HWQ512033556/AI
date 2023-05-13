
#include "opencv2/opencv.hpp"

class Queue {
   public:
    Queue(int maxlen);
    ~Queue();

   public:
    void push(cv::Point loc);
    cv::Point pop();
    bool is_empty();

   private:
    cv::Point *contents;
    int head;
    int tail;
};
