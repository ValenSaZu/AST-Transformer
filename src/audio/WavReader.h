#ifndef WAVREADER_H
#define WAVREADER_H

#include <vector>
#include <string>
#include <stdexcept>
#include <sndfile.h>

using namespace std;

class WavReader {
public:
    static tuple<vector<float>, int, int> load(const string& filepath) {
        SF_INFO sfinfo;
        sfinfo.format = 0;
        
        SNDFILE* file = sf_open(filepath.c_str(), SFM_READ, &sfinfo);
        if (!file) {
            throw runtime_error("Cannot open WAV file: " + string(sf_strerror(NULL)));
        }
        
        // Verificar formato soportado
        if (!isFormatSupported(sfinfo)) {
            sf_close(file);
            throw runtime_error("Unsupported WAV format");
        }
        
        // Leer todos los samples
        vector<float> buffer(static_cast<size_t>(sfinfo.frames * sfinfo.channels));
        sf_count_t framesRead = sf_readf_float(file, buffer.data(), sfinfo.frames);
        
        if (framesRead != sfinfo.frames) {
            sf_close(file);
            throw runtime_error("Error reading WAV file: incomplete read");
        }
        
        sf_close(file);
        
        return {buffer, sfinfo.samplerate, sfinfo.channels};
    }

private:
    static bool isFormatSupported(const SF_INFO& sfinfo) {
        return (sfinfo.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_WAV ||
               (sfinfo.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_FLAC ||
               (sfinfo.format & SF_FORMAT_TYPEMASK) == SF_FORMAT_OGG;
    }
};

#endif