#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ReLULayer<Dtype>::LayerSetUp(bottom, top);
	// initialize cuDNN
	CUDNN_CHECK(cudnnCreate(&handle_));
	cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
	cudnn::createTensor4dDesc<Dtype>(&top_desc_);
	cudnn::createActivationDescriptor<Dtype>(&activ_desc_, CUDNN_ACTIVATION_RELU);
	handles_setup_ = true;
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ReLULayer<Dtype>::Reshape(bottom, top);

	const int bottom_dim = bottom[0]->num_axes() < 4 ? 4 : bottom[0]->num_axes();
	int* bottom_shape = new int[bottom_dim];
	for (int i = 0; i < bottom_dim; ++i) {
		bottom_shape[i] = 1;
	}
	for (int i = 0; i < bottom[0]->num_axes(); ++i) {
		bottom_shape[i] = bottom[0]->shape(i);
	}
	int* bottom_stride = new int[bottom_dim];
	bottom_stride[bottom_dim - 1] = 1;
	for (int i = bottom_dim - 2; i >= 0; --i) {
		bottom_stride[i] = bottom_shape[i + 1] * bottom_stride[i + 1];
	}

	if (bottom_dim > 4)
	{
		cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom_dim, bottom_shape, bottom_stride);
	}
	else
	{
		cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom_shape[0], bottom_shape[1], bottom_shape[2], bottom_shape[3]);
	}

	delete[]bottom_shape;
	delete[]bottom_stride;

	const int top_dim = top[0]->num_axes() < 4 ? 4 : top[0]->num_axes();
	int* top_shape = new int[top_dim];
	for (int i = 0; i < top_dim; ++i) {
		top_shape[i] = 1;
	}
	for (int i = 0; i < top[0]->num_axes(); ++i) {
		top_shape[i] = top[0]->shape(i);
	}
	int* top_stride = new int[top_dim];
	top_stride[top_dim - 1] = 1;
	for (int i = top_dim - 2; i >= 0; --i) {
		top_stride[i] = top_shape[i + 1] * top_stride[i + 1];
	}

	if (top_dim > 4)
	{
		cudnn::setTensorNdDesc<Dtype>(&top_desc_, top_dim, top_shape, top_stride);
	}
	else
	{
		cudnn::setTensor4dDesc<Dtype>(&top_desc_, top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
	}

	delete[]top_shape;
	delete[]top_stride;
}

template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer() {
	// Check that handles have been setup before destroying.
	if (!handles_setup_) { return; }

	cudnnDestroyTensorDescriptor(this->bottom_desc_);
	cudnnDestroyTensorDescriptor(this->top_desc_);
	cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS(CuDNNReLULayer);

}  // namespace caffe
#endif
