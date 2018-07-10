#ifndef CAFFE_MHD_SLICE_DATA_LAYER_HPP_
#define CAFFE_MHD_SLICE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MHDSliceDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::Image<Dtype, 2> ImageSliceType;
  typedef itk::Image<char, 2> LabelSliceType;

  struct SliceRecord {
	  std::string file_name_;
	  std::vector<int> label_exist_;
  };

  struct ImageRecord {
	  ~ImageRecord()
	  {
		  for (std::vector<SliceRecord*>::iterator it = slice_.begin(); it != slice_.end(); ++it) {
			  if (NULL != *it) {
				  delete *it;
				  *it = NULL;
			  }
		  }
		  slice_.clear();

	  }
	  std::string file_name_;
	  int direction_;
	  int slice_size_[2];
	  double slice_spacing_[2];
	  int length_;
	  int seek_;
	  std::vector<SliceRecord*> slice_;
  };

  explicit MHDSliceDataLayer(const LayerParameter& param)
	  : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MHDSliceDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MHDSliceData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  shared_ptr<Caffe::RNG> ind_rng_;
  shared_ptr<Caffe::RNG> trans_rng_;
  vector<ImageRecord*> lines_;
  int contour_num_;
  int lines_id_;
  bool output_roi_;
};

}  // namespace caffe

#endif  // CAFFE_MHD_SLICE_DATA_LAYER_HPP_
