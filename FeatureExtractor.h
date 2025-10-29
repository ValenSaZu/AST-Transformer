//
// Created by Amara Barrera

// Módulo: FeatureExtractor (en CUDA y C++)

// ¿Qué hace?: Transforma el audio ya limpio en una representación visual del sonido (espectrograma)

//FeatureExtractor.h
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//Macros de CUDA para chequeo de errores
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = (call); \
    if (err != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error " << stat << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Estructura que indica la configuración inicial del módulo
struct FeatureExtractorConfig {
    int sample_rate = 16000;
    int fft_size = 512;
    int win_length = 400; // largo de ventana (samples)
    int hop_length = 160;
    int n_mels = 128;
    float f_min = 0.0f;
    float f_max = 8000.0f;
    bool center = false;
    float top_db = 80.0f;
    int pad_time_to_multiple = 16;
};

struct SpectrogramOutput {
    float* d_mel; // puntero en device - GPU
    float* d_frame_mask; // puntero en device - GPU
    int n_mels;
    int n_raw_frames;
    int n_padded_frames;

    //vector para el debugg en CPU
    vector<float> mel_db_host;
    vector<float> frame_mask_host;
};

// Funciones en el CPU
inline float hz_to_mel(float hz){
    return 2595.0f * log10f(1.0f + hz / 700.0f);}

inline float mel_to_hz(float mel){
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);}

inline int ceil_div(int a, int b){
    return (a + b - 1) / b;}

// KERNELS DE CUDA
// Kernel para configuración de la ventana Hann
__global__ void kernel_hann_window(const float* audio, float* windowed_frames, const float* window, int n_frames, int win_length, int fft_size, int hop_length, int waveform_len){
    
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if(frame_idx >= n_frames) return;
    
    int frame_offset = frame_idx * fft_size;
    int audio_start = frame_idx * hop_length;
    if(sample_idx < win_length) {
        int audio_idx = audio_start + sample_idx;
        float v = 0.0f;
        if(audio_idx < waveform_len) v = audio[audio_idx];
        windowed_frames[frame_offset + sample_idx] = v * window[sample_idx];
    }
    
    else if(sample_idx < fft_size)
        windowed_frames[frame_offset + sample_idx] = 0.0f;
}

// Kernel de magnitud al cuadrado para después aplical Mel Filterbank (matriz de filtros que convierte frecuencias lineales a escala mel)
__global__ void kernel_magnitude_squared(const cufftComplex* stft, float* power_spec, int n_frames, int n_freqs){
    
    int frame_idx = blockIdx.x;
    int freq_idx = threadIdx.x;
    
    if (frame_idx >= n_frames || freq_idx >= n_freqs) return;
    
    int idx = frame_idx * n_freqs + freq_idx;
    float real = stft[idx].x;
    float imag = stft[idx].y;
    power_spec[idx] = real * real + imag * imag;
}

// Kernel 1 para normalización
__global__ void kernel_power_to_db_step1(float* mel_spec, int size, float eps){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    mel_spec[idx] = 10.0f * log10f(mel_spec[idx] + eps);
}

// Kernel 2 para normalización
__global__ void kernel_power_to_db_step2(float* mel_spec, int size, float max_val, float top_db){
   
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //fórmula de CUDA para hallar indices globales
    
    if(idx >= size) return;
    
    float v = mel_spec[idx] - max_val;
    float min_val = -top_db;
    if(v < min_val)
        v = min_val;
    mel_spec[idx] = v;
}

// Kernel que agrega ceros a la matriz para que sea compatible con el batch
__global__ void kernel_pad_time(const float* input, float* output, int n_mels, int n_in, int n_out) {
    int mel_idx = blockIdx.x;
    int frame_idx = threadIdx.x + blockIdx.y * blockDim.x;
    if (mel_idx >= n_mels || frame_idx >= n_out) return;
    int out_idx = mel_idx * n_out + frame_idx;
    if (frame_idx < n_in) {
        int in_idx = mel_idx * n_in + frame_idx;
        output[out_idx] = input[in_idx];
    } else {
        output[out_idx] = 0.0f;
    }
}

// CLASE FEATURE EXTRACTOR
class FeatureExtractor {
public:
    // Constructor
    FeatureExtractor(const FeatureExtractorConfig& cfg = FeatureExtractorConfig()):
    config(cfg),
    initialized(false),
    fft_plan(0),
    cublas_handle(nullptr),
    d_window(nullptr),
    d_mel_filterbank(nullptr),
    d_windowed_frames(nullptr),
    d_stft(nullptr),
    d_power_spec(nullptr),
    d_mel_spec(nullptr),
    d_mel_spec_padded(nullptr),
    d_frame_mask(nullptr){
        n_freqs = config.fft_size/2 + 1;
        initialize();
    }

    // Destructor
    ~FeatureExtractor() {
        cleanup();
    }

    // Método principal - recibe waveform en host (float*), longitud, y opcional mask host
    SpectrogramOutput extract(const float* h_waveform, int waveform_length, const float* h_sample_mask = nullptr) {
        SpectrogramOutput output{};
        //1. calcula cuantas frames se pueden extraer del audio
        int n_frames = compute_num_frames(waveform_length);

        //2. reserva memoria en GPU y copia el audio original del CPU
        float* d_waveform = nullptr;
        CUDA_CHECK(cudaMalloc(&d_waveform, waveform_length * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_waveform, h_waveform, waveform_length * sizeof(float), cudaMemcpyHostToDevice));

        //3. crea buffers temporales donde se guardarán ventanas, STFT, espectograma de potencia
        allocate_buffers(n_frames);

        //4. cada frame de audio es multiplicado por una ventana Hann que suaviza border para evitar saltos bruscos
        {
            dim3 grid(n_frames);
            dim3 block(config.fft_size);
            kernel_hann_window<<<grid, block>>>(d_waveform, d_windowed_frames, d_window, n_frames, config.win_length, config.fft_size, config.hop_length, waveform_length);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        //5. aplica Transformada rápida de Fourier
        CUFFT_CHECK(cufftPlan1d(&fft_plan, config.fft_size, CUFFT_R2C, n_frames));
        CUFFT_CHECK(cufftExecR2C(fft_plan, d_windowed_frames, d_stft));
        CUDA_CHECK(cudaDeviceSynchronize());

        //6. calcula magnitud al cuadrado convirtiendo cada valor complejo de la FFT en su potencia (es el espectograma lineal de energí sin escala log)
        {
            dim3 grid(n_frames);
            dim3 block( (n_freqs <= 1024) ? n_freqs : 1024 );
            kernel_magnitude_squared<<<grid, block>>>(d_stft, d_power_spec, n_frames, n_freqs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        //7. aplica el filtro Mel usando cuBLAS
            //multiplica el espectograma de potencia (en frecuencias lineales) por la matriz de filtros Mel para obtener el Mel-espectograma
            //mel_spec = filterbank [n_mels x n_freqs] * power_spec^T [n_freqs x n_frames]
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     n_frames, config.n_mels, n_freqs,
                                     &alpha,
                                     d_power_spec, n_freqs,
                                     d_mel_filterbank, n_freqs,
                                     &beta,
                                     d_mel_spec, n_frames));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        //8. convierte a decibelios (log) - aplica el logaritmo 10*log10(x) para escalar el espectograma
            //Aquí se obtiene el log-mel espectograma
        {
            int size = config.n_mels * n_frames;
            int threads = 256;
            int blocks = ceil_div(size, threads);
            kernel_power_to_db_step1<<<blocks, threads>>>(d_mel_spec, size, 1e-10f);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            //encuentra el max con cublasIsamax
            int max_idx = 0;
            CUBLAS_CHECK(cublasIsamax(cublas_handle, size, d_mel_spec, 1, &max_idx));
            float max_val = 0.0f;
            
            if(max_idx > 0)
                CUDA_CHECK(cudaMemcpy(&max_val, d_mel_spec + (max_idx - 1), sizeof(float), cudaMemcpyDeviceToHost));
            
            
            kernel_power_to_db_step2<<<blocks, threads>>>(d_mel_spec, size, max_val, config.top_db);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        //9. genera la máscara de frames - esta máscara marca que frames son válidos (1) y cuales son padding (0)
        generate_frame_mask(h_sample_mask, waveform_length, n_frames);

        //10. rellena el eje temporal - si el num de frames no es múltiplo de 16 rellena con 0 hasta llegar al múltiplo
        int n_frames_padded = pad_time_axis(n_frames);

        //prepara el output (espectograma mel y la máscara)
        output.d_mel = d_mel_spec_padded;
        output.d_frame_mask = d_frame_mask;
        output.n_mels = config.n_mels;
        output.n_raw_frames = n_frames;
        output.n_padded_frames = n_frames_padded;

        //liberar memoria en GPU
        CUDA_CHECK(cudaFree(d_waveform));

        return output;
    }

    //Copiar buffers del device (GPU) al host (CPU)
    void copy_to_host(SpectrogramOutput& output){
        
        int mel_size = output.n_mels * output.n_padded_frames;
        output.mel_db_host.resize(mel_size);
        output.frame_mask_host.resize(output.n_padded_frames);
        
        CUDA_CHECK(cudaMemcpy(output.mel_db_host.data(), output.d_mel, mel_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output.frame_mask_host.data(), output.d_frame_mask, output.n_padded_frames * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    FeatureExtractorConfig config;
    bool initialized;

    //Planificadores de la Transformada Rapida de Fouries (cufftHandle) y de la libreria de algebra lineal (cublasHandle_t)
    cufftHandle fft_plan;
    cublasHandle_t cublas_handle;

    //Buffers del device (GPU)
    float* d_window;
    float* d_mel_filterbank;
    float* d_windowed_frames;
    cufftComplex* d_stft;
    float* d_power_spec;
    float* d_mel_spec;
    float* d_mel_spec_padded;
    float* d_frame_mask;

    int n_freqs;

    //Inicializar
    void initialize(){
        if(initialized) return;
        //Crea ventana Hann del host (CPU) al device (GPU)
        vector<float> h_window(config.win_length);
        for (int i = 0; i < config.win_length; ++i)
            h_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (config.win_length - 1)));
        
        CUDA_CHECK(cudaMalloc(&d_window, config.win_length * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_window, h_window.data(), config.win_length * sizeof(float), cudaMemcpyHostToDevice));

        //Crea filtro mel del host (CPU) al device (GPU)
        float f_max = config.f_max > 0 ? config.f_max : config.sample_rate / 2.0f;
        float mel_min = hz_to_mel(config.f_min);
        float mel_max = hz_to_mel(f_max);
        
        vector<float> mel_points(config.n_mels + 2);
        for(int i = 0; i < config.n_mels + 2; ++i) mel_points[i] = mel_min + (mel_max - mel_min) * i / (config.n_mels + 1);
        
        vector<float> hz_points(config.n_mels + 2);
        for(int i = 0; i < config.n_mels + 2; ++i) hz_points[i] = mel_to_hz(mel_points[i]);
       
        vector<int> bin_points(config.n_mels + 2);
        for(int i = 0; i < config.n_mels + 2; ++i)
            bin_points[i] = static_cast<int>(floorf((config.fft_size + 1) * hz_points[i] / config.sample_rate));
        
        vector<float> h_filterbank(config.n_mels * n_freqs, 0.0f);
        for(int m = 1; m <= config.n_mels; ++m){
            int f_left = bin_points[m - 1];
            int f_center = bin_points[m];
            int f_right = bin_points[m + 1];
            if (f_left < f_center){
                for (int f = f_left; f < f_center && f < n_freqs; ++f)
                    h_filterbank[(m - 1) * n_freqs + f] = float(f - f_left) / float(f_center - f_left);
            }
            if(f_center < f_right){
                for(int f = f_center; f < f_right && f < n_freqs; ++f)
                    h_filterbank[(m - 1) * n_freqs + f] = float(f_right - f) / float(f_right - f_center);
            }
        }
        
        //Normaliza cada filtro mel según la escala Slaney
        //Ajusta la ganancia de cada filtro para que todos tengan una energía comparable
        for(int m = 0; m < config.n_mels; ++m){
            float denom = hz_points[m + 2] - hz_points[m];
            if (denom <= 0) continue;
            float enorm = 2.0f / denom;
            for (int f = 0; f < n_freqs; ++f) h_filterbank[m * n_freqs + f] *= enorm;
        }
        
        //crea memoria en el GPU con cudaMalloc y copia la matriz del filtro mel
        CUDA_CHECK(cudaMalloc(&d_mel_filterbank, config.n_mels * n_freqs * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mel_filterbank, h_filterbank.data(), config.n_mels * n_freqs * sizeof(float), cudaMemcpyHostToDevice));

        //crea cuBLAS que permite ejecutar operacines de algebra lineal en el GPU
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        initialized = true;
    }

    //Liberar memoria en GPU
    void cleanup(){
        if(!initialized) return;
        if(fft_plan) cufftDestroy(fft_plan);
        if(cublas_handle) cublasDestroy(cublas_handle);
        if(d_window) cudaFree(d_window);
        if(d_mel_filterbank) cudaFree(d_mel_filterbank);
        if(d_windowed_frames) cudaFree(d_windowed_frames);
        if(d_stft) cudaFree(d_stft);
        if(d_power_spec) cudaFree(d_power_spec);
        if(d_mel_spec) cudaFree(d_mel_spec);
        if(d_mel_spec_padded) cudaFree(d_mel_spec_padded);
        if(d_frame_mask) cudaFree(d_frame_mask);
        initialized = false;
    }

    //FUNCIONES AYUDANTES DEL CPU
    int compute_num_frames(int waveform_length) const{
        if(waveform_length < config.win_length) return 1;
        return 1 + (waveform_length - config.win_length) / config.hop_length;
    }

    void allocate_buffers(int n_frames){
        CUDA_CHECK(cudaMalloc(&d_windowed_frames, n_frames * config.fft_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_stft, n_frames * n_freqs * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_power_spec, n_frames * n_freqs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mel_spec, config.n_mels * n_frames * sizeof(float)));
        //d_mel_spec_padded se reservará en pad_time_axis porque depende de si el num de frames necesita padding
    }

    //Genera la máscara de frames - indicando que partes del audio son validas
    void generate_frame_mask(const float* h_sample_mask, int waveform_length, int n_frames){
        vector<float> h_frame_mask(n_frames, 0.0f);
        if(h_sample_mask == nullptr)
            fill(h_frame_mask.begin(), h_frame_mask.end(), 1.0f);
        
        else {
            int n_real = 0;
            for(int i = 0; i < waveform_length; ++i) if (h_sample_mask[i] > 0.0f) ++n_real;
            if(n_real <= 0) n_real = 0;
            else if(n_real < config.win_length) n_real = 1;
            else n_real = 1 + (n_real - config.win_length) / config.hop_length;
            int n_valid = std::min(n_real, n_frames);
            for(int i = 0; i < n_valid; ++i) h_frame_mask[i] = 1.0f;
        }
        CUDA_CHECK(cudaMalloc(&d_frame_mask, n_frames * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_frame_mask, h_frame_mask.data(), n_frames * sizeof(float), cudaMemcpyHostToDevice));
    }

    //Ajusta (pad) el eje temporal del espectograma para que el num de frames sea múltiplo del valor fijo en configuration
    int pad_time_axis(int n_frames_raw){
        int n_frames_padded;
        if config.pad_time_to_multiple > 0)
            n_frames_padded = ceil_div(n_frames_raw, config.pad_time_to_multiple) * config.pad_time_to_multiple;
        else
            n_frames_padded = n_frames_raw;

        if(n_frames_padded == n_frames_raw){
            d_mel_spec_padded = d_mel_spec;
            return n_frames_padded;
        }

        CUDA_CHECK(cudaMalloc(&d_mel_spec_padded, config.n_mels * n_frames_padded * sizeof(float)));
        dim3 block(256);
        dim3 grid(config.n_mels, ceil_div(n_frames_padded, 256));
        kernel_pad_time<<<grid, block>>>(d_mel_spec, d_mel_spec_padded, config.n_mels, n_frames_raw, n_frames_padded);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        //máscara de frames
        float* d_frame_mask_padded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_frame_mask_padded, n_frames_padded * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_frame_mask_padded, d_frame_mask, n_frames_raw * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_frame_mask_padded + n_frames_raw, 0, (n_frames_padded - n_frames_raw) * sizeof(float)));
        cudaFree(d_frame_mask);
        d_frame_mask = d_frame_mask_padded;

        return n_frames_padded;
    }
};

#endif
