#include <algorithm>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void DiceLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
  const int spatial_dim = bottom[0]->count() / num / channel;

  std::vector<int> shapes(1, num * channel);
  intersection_.Reshape(shapes);
  union_.Reshape(shapes);
  class_exist_ = new bool[num * channel];
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  const int count = bottom[0]->count();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
  const int spatial_dim = count / num / channel;
  buffer_.ReshapeLike(*bottom[0]);
  std::vector<int> mask_shape(1, spatial_dim);
  ones_mask_.Reshape(mask_shape);
  Dtype* ones_mask_data = ones_mask_.mutable_cpu_data();
  for (int i = 0; i < spatial_dim; ++i) {
	ones_mask_data[i] = 1.0;
  }
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  const int count = bottom[0]->count();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
  const int spatial_dim = count / num / channel;
  Dtype* label_data = new Dtype[num * channel * spatial_dim];
  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      Dtype max_score = -1;
      int max_score_cls = 0;
      for (int c = 0; c < channel; ++c) {
        if (max_score < prob_data[n * channel * spatial_dim + c * spatial_dim + s]) {
          max_score = prob_data[n * channel * spatial_dim + c * spatial_dim + s];
          max_score_cls = c;
        }
      }
      for (int c = 0; c < channel; ++c) {
        label_data[n * channel * spatial_dim + c * spatial_dim + s] = (max_score_cls == c);
      }
    }
  }

  int exist_class_num = 0;
  Dtype* intersection_data = intersection_.mutable_cpu_data();
  Dtype* union_data = union_.mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channel; ++c) {
      Dtype label_sum = 0;
	  Dtype gt_sum = 0;
	  intersection_data[n * channel + c] = 0;
	  union_data[n * channel + c] = 0;
      class_exist_[n * channel + c] = false;
      for (int s = 0; s < spatial_dim; ++s) {
        const Dtype label_value = 
			label_data[n * channel * spatial_dim + c * spatial_dim + s];
		const Dtype gt_value =
			(gt_data[n * spatial_dim + s] == c);

		intersection_data[n * channel + c] +=
			label_value * gt_value;
        
		union_data[n * channel + c] +=
			label_value * label_value + gt_value * gt_value;

        if (c > 0) {
          label_sum += label_value;
		  gt_sum += gt_value;
        }
      }
	  union_data[n * channel + c] += 0.00001;
	  if (gt_sum > 0) {
        class_exist_[n * channel + c] = true;
        exist_class_num++;
      }
    }
  }
  
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int n = 0; n < num; ++n) {
    for (int j = 0; j < channel; ++j) {
      if (class_exist_[n * channel + j]) {
        loss[0] += 2 * intersection_data[n * channel + j] / union_data[n * channel + j];
      }
    }
  }
  if (exist_class_num > 0) {
    loss[0] = loss[0] / exist_class_num;
  }
  LOG(INFO) << "Average dice(CPU) = " << loss[0];

  delete[]label_data;
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* prob_data = bottom[0]->cpu_data();
    const Dtype* gt_data = bottom[1]->cpu_data();
	const Dtype* intersection_data = intersection_.cpu_data();
	const Dtype* union_data = union_.cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
    const int spatial_dim = count / num / channel;

	memset(bottom_diff, 0, sizeof(Dtype) * count);
    for (int n = 0; n < num; ++n) {
      for (int s = 0; s < spatial_dim; ++s) {
        for (int c = 1; c < channel; ++c) {
          if (class_exist_[n * channel + c]) {
            const Dtype prob_value = 
				prob_data[n * channel * spatial_dim + c * spatial_dim + s];
			const Dtype gt_value =
				(gt_data[n * spatial_dim + s] == c);
			const Dtype intersection_value = intersection_data[n * channel + c];
			const Dtype union_value = union_data[n * channel + c];
			const Dtype diff =
              2 * (gt_value * union_value / (union_value * union_value) -
			  2 * prob_value * intersection_value / (union_value * union_value));

            bottom_diff[n * channel * spatial_dim + c * spatial_dim + s] -= diff;              
            bottom_diff[n * channel * spatial_dim + 0 * spatial_dim + s] += diff;
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(DiceLossLayer);
REGISTER_LAYER_CLASS(DiceLoss);

}  // namespace caffe