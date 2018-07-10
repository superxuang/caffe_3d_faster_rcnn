#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DiceLossForwardGPU(const int nthreads, 
	      const int channel, const int spatial_dim, 
          const Dtype* prob_data, const Dtype* gt_data, 
		  Dtype* intersection_buffer, Dtype* union_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
	const int s = index % spatial_dim;
	Dtype max_score = -1;
	int max_score_cls = 0;
    for (int c = 0; c < channel; ++c) {
      if (max_score < prob_data[n * channel * spatial_dim + c * spatial_dim + s])
      {
        max_score = prob_data[n * channel * spatial_dim + c * spatial_dim + s];
        max_score_cls = c;
      }
    }
    for (int c = 0; c < channel; ++c) {
	  const Dtype label_value = (max_score_cls == c);
	  const Dtype gt_value = (gt_data[n * spatial_dim + s] == c);
	  intersection_buffer[n * channel * spatial_dim + c * spatial_dim + s] = 
		  label_value * gt_value;
	  union_buffer[n * channel * spatial_dim + c * spatial_dim + s] = 
		  label_value * label_value + gt_value * gt_value;
    }
  }
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* prob_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->shape(0);
  const int channel = bottom[0]->shape(1);
  const int spatial_dim = count / num / channel;
  const int nthreads = count / channel;
  Dtype* intersection_buffer = buffer_.mutable_gpu_data();
  Dtype* union_buffer = buffer_.mutable_gpu_diff();
  const Dtype* ones = ones_mask_.gpu_data();
  DiceLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, channel, spatial_dim, 
	  prob_data, gt_data, intersection_buffer, union_buffer);
  Dtype* intersection_data = intersection_.mutable_cpu_data();
  Dtype* union_data = union_.mutable_cpu_data();
  for (int n = 0; n < num * channel; ++n) {
	caffe_gpu_dot(spatial_dim, intersection_buffer + n * spatial_dim, 
	    ones, intersection_data + n);
	caffe_gpu_dot(spatial_dim, union_buffer + n * spatial_dim,
		ones, union_data + n);
	union_data[n] += 0.00001;
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int n = 0; n < num * channel; ++n) {
    loss[0] += 2 * intersection_data[n] / union_data[n];
  }
  loss[0] = loss[0] / num / channel;
  LOG(INFO) << "Average dice(GPU) = " << loss[0];
}

template <typename Dtype>
__global__ void DiceLossBackwardGPU(const int nthreads, 
	      const int channel, const int spatial_dim, 
	      const Dtype* prob_data, const Dtype* gt_data, 
	      const Dtype* intersection_data, const Dtype* union_data, 
		  Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
	const int s = index % spatial_dim;
    for (int c = 1; c < channel; ++c) {
      const Dtype prob_value = prob_data[n * channel * spatial_dim + c * spatial_dim + s];
	  const Dtype gt_value = (gt_data[n * spatial_dim + s] == c);
	  const Dtype union_value = union_data[n * channel + c];
	  const Dtype intersection_value = intersection_data[n * channel + c];
	  const Dtype diff =
		  2 * (gt_value * union_value / (union_value * union_value) -
		  2 * prob_value * intersection_value / (union_value * union_value));

	  bottom_diff[n * channel * spatial_dim + c * spatial_dim + s] -= diff;
	  bottom_diff[n * channel * spatial_dim + 0 * spatial_dim + s] += diff;
	}
  }
}

template <typename Dtype>
void DiceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	const int count = bottom[0]->count();
    const int num = bottom[0]->shape(0);
    const int channel = bottom[0]->shape(1);
    const int spatial_dim = count / num / channel;
	const int nthreads = count / channel;
	const Dtype* prob_data = bottom[0]->gpu_data();
    const Dtype* gt_data = bottom[1]->gpu_data();
	const Dtype* intersection_data = intersection_.gpu_data();
	const Dtype* union_data = union_.gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	caffe_gpu_memset(count * sizeof(Dtype), 0, bottom_diff);
	DiceLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, channel, spatial_dim, 
		prob_data, gt_data, intersection_data, union_data, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DiceLossLayer);

}  // namespace caffe
