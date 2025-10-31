#ifndef AUDIOLOADER_H
#define AUDIOLOADER_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <memory>
#include "WavReader.h"
#include <samplerate.h>
#include <iostream>

using namespace std;

/**
 * AudioLoader - Carga y preprocesamiento de audio
 * 
 * Basado en la implementación Python de María Belén
 * Referencia: python/audioloader.py
 * 
 * Autora: María Belén (implementación Python original)
 * Adaptación C++: Camila Valentina Salazar Zuñiga
 */

struct AudioLoaderConfig {
    int sampleRate = 16000;
    float durationSec = 10.0f;
    bool mono = true;
    string normalize = "peak";
    float rmsTargetDbfs = -20.0f;
    string padSide = "end";
    string crop = "center";
};

struct AudioLoadResult {
    vector<float> waveform;
    vector<float> mask;
    int originalLength = 0;
    int targetLength = 0;
    int originalSampleRate = 0;
    int targetSampleRate = 0;
    int channels = 1;
    string filepath;
};

class AudioLoader {
private:
    AudioLoaderConfig config;
    int targetSamples;
    mt19937 rng;

public:
    AudioLoader(const AudioLoaderConfig& cfg = AudioLoaderConfig()) 
        : config(cfg), rng(random_device{}()) {
        targetSamples = static_cast<int>(config.sampleRate * config.durationSec);
    }

    AudioLoadResult load(const string& filepath) {
        AudioLoadResult result;
        result.filepath = filepath;
        
        try {
            // 1. Cargar archivo WAV con libsndfile
            auto [audio, sampleRate, channels] = WavReader::load(filepath);
            result.originalSampleRate = sampleRate;
            result.targetSampleRate = config.sampleRate;
            result.originalLength = static_cast<int>(audio.size());
            result.channels = channels;
            
            cout << "Loaded: " << filepath 
                    << " (" << sampleRate << " Hz, " << channels << " channels, "
                    << audio.size() << " samples)" << endl;
            
            // 2. Convertir a mono si es necesario
            if (config.mono && channels > 1) {
                audio = toMono(audio, channels);
                result.channels = 1;
                cout << "Converted to mono: " << audio.size() << " samples" << endl;
            }
            
            // 3. Resamplear si es necesario
            if (sampleRate != config.sampleRate) {
                cout << "Resampling from " << sampleRate << " Hz to " 
                        << config.sampleRate << " Hz" << endl;
                audio = resampleAudio(audio, sampleRate, config.sampleRate);
            }
            
            // 4. Normalizar
            if (config.normalize == "peak") {
                audio = normalizePeak(audio);
                cout << "Applied peak normalization" << endl;
            } else if (config.normalize == "rms") {
                audio = normalizeRms(audio, config.rmsTargetDbfs);
                cout << "Applied RMS normalization to " << config.rmsTargetDbfs << " dBFS" << endl;
            }
            
            // 5. Aplicar padding o crop
            auto [paddedAudio, mask] = padOrCrop(audio, targetSamples, config.padSide, config.crop);
            
            result.waveform = paddedAudio;
            result.mask = mask;
            result.targetLength = targetSamples;
            
            cout << "Final waveform: " << result.waveform.size() 
                    << " samples, mask: " << result.mask.size() << " elements" << endl;
            
        } catch (const exception& e) {
            throw runtime_error("Error loading audio file '" + filepath + "': " + e.what());
        }
        
        return result;
    }

private:
    // Función auxiliar para calcular RMS de forma segura
    float safeRms(const vector<float>& x) {
        if (x.empty()) return 0.0f;
        
        double sum = 0.0;
        for (float val : x) {
            sum += static_cast<double>(val) * val;
        }
        return static_cast<float>(sqrt(sum / x.size() + 1e-12));
    }
    
    // Convertir dBFS a escala lineal
    float dbfsToLinear(float dbfs) {
        return pow(10.0f, dbfs / 20.0f);
    }
    
    // Convertir a mono (promedio de canales)
    vector<float> toMono(const vector<float>& audio, int channels) {
        if (channels <= 1) return audio;
        
        size_t numFrames = audio.size() / channels;
        vector<float> mono(numFrames);
        
        for (size_t i = 0; i < numFrames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch) {
                sum += audio[i * channels + ch];
            }
            mono[i] = sum / channels;
        }
        
        return mono;
    }
    
    // Resampleo de alta calidad con libsamplerate
    vector<float> resampleAudio(const vector<float>& x, int srcSr, int dstSr) {
        if (srcSr == dstSr) return x;
        
        double ratio = static_cast<double>(dstSr) / srcSr;
        size_t outputSize = static_cast<size_t>(x.size() * ratio + 0.5);
        
        vector<float> output(outputSize);
        
        SRC_DATA src_data;
        src_data.data_in = const_cast<float*>(x.data());
        src_data.input_frames = x.size();
        src_data.data_out = output.data();
        src_data.output_frames = outputSize;
        src_data.src_ratio = ratio;
        
        int error = src_simple(&src_data, SRC_SINC_MEDIUM_QUALITY, 1);
        if (error) {
            throw runtime_error("Resampling failed: " + string(src_strerror(error)));
        }
        
        output.resize(src_data.output_frames_gen);
        return output;
    }
    
    // Normalización peak
    vector<float> normalizePeak(const vector<float>& x) {
        if (x.empty()) return x;
        
        float peak = 0.0f;
        for (float val : x) {
            peak = max(peak, abs(val));
        }
        
        if (peak < 1e-9f) return x;
        
        vector<float> result(x.size());
        float scale = 0.99f / peak;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = x[i] * scale;
        }
        
        return result;
    }
    
    // Normalización RMS
    vector<float> normalizeRms(const vector<float>& x, float targetDbfs) {
        if (x.empty()) return x;
        
        float currentRms = safeRms(x);
        if (currentRms < 1e-9f) return x;
        
        float targetLinear = dbfsToLinear(targetDbfs);
        float scale = targetLinear / currentRms;
        
        vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = clamp(x[i] * scale, -1.0f, 1.0f);
        }
        
        return result;
    }
    
    // Padding o crop del audio
    pair<vector<float>, vector<float>> padOrCrop(const vector<float>& x, int nTarget, const string& padSide, const string& crop) {
        vector<float> y(nTarget, 0.0f);
        vector<float> mask(nTarget, 1.0f);
        
        int n = static_cast<int>(x.size());
        
        if (n == nTarget) {
            copy(x.begin(), x.end(), y.begin());
            return {y, mask};
        }
        
        // Crop si es más largo
        if (n > nTarget) {
            int start = 0;
            if (crop == "start") {
                start = 0;
            } else if (crop == "random") {
                uniform_int_distribution<int> dist(0, n - nTarget);
                start = dist(rng);
            } else { // "center"
                start = (n - nTarget) / 2;
            }
            
            copy(x.begin() + start, x.begin() + start + nTarget, y.begin());
            return {y, mask};
        }
        
        // Padding si es más corto
        if (padSide == "both") {
            int totalPad = nTarget - n;
            int left = totalPad / 2;
            int right = totalPad - left;
            
            copy(x.begin(), x.end(), y.begin() + left);
            fill(mask.begin(), mask.begin() + left, 0.0f);
            fill(mask.begin() + left + n, mask.end(), 0.0f);
            
        } else if (padSide == "start") {
            copy(x.begin(), x.end(), y.begin() + (nTarget - n));
            fill(mask.begin(), mask.begin() + (nTarget - n), 0.0f);
            
        } else { // "end"
            copy(x.begin(), x.end(), y.begin());
            fill(mask.begin() + n, mask.end(), 0.0f);
        }
        
        return {y, mask};
    }
};

#endif