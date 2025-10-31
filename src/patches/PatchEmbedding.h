#ifndef PATCH_EMBEDDING_H
#define PATCH_EMBEDDING_H

/**
 * PatchEmbedding - Divide el espectrograma MEL en parches y los proyecta a embeddings
 * 
 * Autora: Camila Valentina Salazar Zuñiga
 * 
 * Funcionalidades:
 * - Divide el espectrograma MEL en parches de tamaño fijo (16x16)
 * - Aplana cada parche y lo proyecta con una matriz de pesos aprendible
 * - Agrega embeddings posicionales para preservar información temporal
 * - Genera máscara de parches válidos
 */

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

using namespace std;

struct PatchEmbeddingConfig {
    int patchSize = 16;           // Tamaño del parche (pF x pT)
    int hiddenSize = 768;         // Dimensión del embedding (típico en transformers)
    int nMelBands = 128;          // Debe coincidir con FeatureExtractor n_mels
    float dropoutRate = 0.1f;     // Tasa de dropout para regularización
    bool usePositionalEmbedding = true;
};

struct PatchEmbeddingOutput {
    vector<float> embeddings;      // [num_patches, hidden_size]
    vector<float> patchMask;       // [num_patches]
    vector<int> patchPositions;    // [num_patches, 2] - posiciones (fila, columna)
    int numPatches;
    int gridHeight;
    int gridWidth;
};

class PatchEmbedding {
private:
    PatchEmbeddingConfig config;
    vector<float> projectionWeights;  // [patchSize*patchSize, hiddenSize]
    vector<float> positionalEmbeddings; // [num_patches, hiddenSize]
    vector<float> clsToken;           // [hiddenSize] - token de clasificación
    mt19937 rng;
    
public:
    /*
    Camila Valentina Salazar Zuñiga
    Constructor de la clase PatchEmbedding
    Parametros: cfg - Configuración para el patch embedding
    Retorna: Instancia de PatchEmbedding inicializada
    */
    PatchEmbedding(const PatchEmbeddingConfig& cfg = PatchEmbeddingConfig()) 
        : config(cfg), rng(random_device{}()) {
        initializeWeights();
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Convierte espectrograma MEL en secuencia de parches con embeddings 
    Parametros:
    melSpectrogram: [n_mels, n_frames] del FeatureExtractor
    frameMask: [n_frames] máscara del FeatureExtractor
    n_mels: número de bandas mel
    n_frames: número de frames temporales
    Retorna: PatchEmbeddingOutput con embeddings y máscaras
    */
    PatchEmbeddingOutput patchify(const vector<float>& melSpectrogram, 
                                 const vector<float>& frameMask,
                                 int n_mels, int n_frames) {
        
        // Validar dimensiones de entrada
        if (melSpectrogram.size() != n_mels * n_frames) {
            throw invalid_argument("Dimensiones del espectrograma no coinciden");
        }
        
        if (n_mels % config.patchSize != 0 || n_frames % config.patchSize != 0) {
            throw invalid_argument("Dimensiones del espectrograma deben ser divisibles por patchSize");
        }
        
        PatchEmbeddingOutput output;
        
        // 1. Calcular grid de parches
        output.gridHeight = n_mels / config.patchSize;
        output.gridWidth = n_frames / config.patchSize;
        output.numPatches = output.gridHeight * output.gridWidth;
        
        cout << "Grid: " << output.gridHeight << " x " << output.gridWidth 
                  << " = " << output.numPatches << " parches" << endl;
        
        // 2. Dividir en parches y generar máscara
        auto patches = extractPatches(melSpectrogram, n_mels, n_frames);
        output.patchMask = createPatchMask(frameMask, n_frames);
        output.patchPositions = computePatchPositions(output.gridHeight, output.gridWidth);
        
        // 3. Proyectar parches a embeddings
        output.embeddings = projectPatches(patches, output.numPatches);
        
        // 4. Agregar embeddings posicionales
        if (config.usePositionalEmbedding) {
            addPositionalEmbeddings(output.embeddings, output.numPatches);
        }
        
        // 5. Agregar token CLS (opcional, para clasificación)
        addClsToken(output);
        
        return output;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Obtiene los pesos de proyección
    Retorna: referencia constante al vector de pesos de proyección
    */
    const vector<float>& getProjectionWeights() const { return projectionWeights; }
    
    /*
    Camila Valentina Salazar Zuñiga
    Obtiene los embeddings posicionales
    Retorna: referencia constante al vector de embeddings posicionales
    */
    const vector<float>& getPositionalEmbeddings() const { return positionalEmbeddings; }
    
private:
    /*
    Camila Valentina Salazar Zuñiga
    Inicializa los pesos del modelo (proyección y embeddings posicionales)
    No recibe parámetros
    No retorna valores
    */
    void initializeWeights() {
        // Inicializar pesos de proyección (Xavier uniform)
        int patchDim = config.patchSize * config.patchSize;
        int totalWeights = patchDim * config.hiddenSize;
        projectionWeights.resize(totalWeights);
        
        float scale = sqrt(6.0f / (patchDim + config.hiddenSize));
        uniform_real_distribution<float> dist(-scale, scale);
        
        for (int i = 0; i < totalWeights; ++i) {
            projectionWeights[i] = dist(rng);
        }
        
        // Inicializar embeddings posicionales
        if (config.usePositionalEmbedding) {
            // Estimamos máximo de parches (ya lo ajustaremos segun veamos)
            int maxPatches = 1024;
            positionalEmbeddings.resize(maxPatches * config.hiddenSize);
            
            normal_distribution<float> normal_dist(0.0f, 0.02f);
            for (int i = 0; i < maxPatches * config.hiddenSize; ++i) {
                positionalEmbeddings[i] = normal_dist(rng);
            }
        }
        
        // Inicializar token CLS
        clsToken.resize(config.hiddenSize);
        normal_distribution<float> normal_dist(0.0f, 0.02f);
        for (int i = 0; i < config.hiddenSize; ++i) {
            clsToken[i] = normal_dist(rng);
        }
        
        cout << "PatchEmbedding inicializado: " 
                  << patchDim << " -> " << config.hiddenSize << "D" << endl;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Extrae parches del espectrograma MEL
    Parametros:
    mel: espectrograma MEL como vector plano [n_mels * n_frames]
    n_mels: número de bandas mel
    n_frames: número de frames temporales
    Retorna: vector de parches, cada parche es un vector de patchSize*patchSize elementos
    */
    vector<vector<float>> extractPatches(const vector<float>& mel, 
                                                  int n_mels, int n_frames) {
        vector<vector<float>> patches;
        int patchDim = config.patchSize * config.patchSize;
        
        for (int gridRow = 0; gridRow < n_mels / config.patchSize; ++gridRow) {
            for (int gridCol = 0; gridCol < n_frames / config.patchSize; ++gridCol) {
                vector<float> patch(patchDim);
                int patchIdx = 0;
                
                // Extraer parche config.patchSize x config.patchSize
                for (int i = 0; i < config.patchSize; ++i) {
                    for (int j = 0; j < config.patchSize; ++j) {
                        int melRow = gridRow * config.patchSize + i;
                        int melCol = gridCol * config.patchSize + j;
                        int idx = melRow * n_frames + melCol;
                        patch[patchIdx++] = mel[idx];
                    }
                }
                
                patches.push_back(patch);
            }
        }
        
        return patches;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Crea máscara de parches basada en la máscara de frames
    Parametros:
    frameMask: máscara de frames [n_frames] donde 1.0 indica frame válido
    n_frames: número de frames temporales
    Retorna: vector de máscaras de parches donde 1.0 indica parche válido
    */
    vector<float> createPatchMask(const vector<float>& frameMask, int n_frames) {
        int numTimePatches = n_frames / config.patchSize;
        vector<float> patchMask(numTimePatches, 0.0f);
        
        // Un parche de tiempo es válido si AL MENOS UN frame en ese bloque es válido
        for (int tPatch = 0; tPatch < numTimePatches; ++tPatch) {
            int startFrame = tPatch * config.patchSize;
            int endFrame = startFrame + config.patchSize;
            
            for (int t = startFrame; t < endFrame && t < frameMask.size(); ++t) {
                if (frameMask[t] > 0.5f) {  // Frame válido
                    patchMask[tPatch] = 1.0f;
                    break;
                }
            }
        }
        
        // Repetir para todas las filas de frecuencia
        vector<float> fullPatchMask;
        int gridHeight = config.nMelBands / config.patchSize;
        for (int fPatch = 0; fPatch < gridHeight; ++fPatch) {
            fullPatchMask.insert(fullPatchMask.end(), patchMask.begin(), patchMask.end());
        }
        
        return fullPatchMask;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Calcula las posiciones de cada parche en el grid
    Parametros:
    gridHeight: número de parches en dimensión frecuencia
    gridWidth: número de parches en dimensión tiempo
    Retorna: vector de posiciones [fila, columna] para cada parche
    */
    vector<int> computePatchPositions(int gridHeight, int gridWidth) {
        vector<int> positions;
        positions.reserve(gridHeight * gridWidth * 2);
        
        for (int i = 0; i < gridHeight; ++i) {
            for (int j = 0; j < gridWidth; ++j) {
                positions.push_back(i);  // fila
                positions.push_back(j);  // columna
            }
        }
        
        return positions;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Proyecta los parches al espacio de embeddings usando multiplicación matricial
    Parametros:
    patches: vector de parches a proyectar
    numPatches: número total de parches
    Retorna: vector de embeddings proyectados [num_patches * hidden_size]
    */
    vector<float> projectPatches(const vector<vector<float>>& patches, 
                                     int numPatches) {
        int patchDim = config.patchSize * config.patchSize;
        vector<float> embeddings(numPatches * config.hiddenSize, 0.0f);
        
        // Multiplicación matriz-vector: patches[P, D] x weights[D, H] = embeddings[P, H]
        for (int p = 0; p < numPatches; ++p) {
            for (int h = 0; h < config.hiddenSize; ++h) {
                float sum = 0.0f;
                for (int d = 0; d < patchDim; ++d) {
                    sum += patches[p][d] * projectionWeights[d * config.hiddenSize + h];
                }
                embeddings[p * config.hiddenSize + h] = sum;
            }
        }
        
        return embeddings;
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Agrega embeddings posicionales a los embeddings de parches
    Parametros:
    embeddings: embeddings de parches a modificar [num_patches * hidden_size]
    numPatches: número total de parches
    No retorna valores (modifica el vector por referencia)
    */
    void addPositionalEmbeddings(vector<float>& embeddings, int numPatches) {
        for (int p = 0; p < numPatches; ++p) {
            for (int h = 0; h < config.hiddenSize; ++h) {
                embeddings[p * config.hiddenSize + h] += 
                    positionalEmbeddings[p * config.hiddenSize + h];
            }
        }
    }
    
    /*
    Camila Valentina Salazar Zuñiga
    Agrega token CLS al inicio de la secuencia de embeddings
    Parametros:
    output: estructura de output a modificar
    No retorna valores (modifica la estructura por referencia)
    */
    void addClsToken(PatchEmbeddingOutput& output) {
        // Agregar token CLS al inicio de la secuencia
        vector<float> newEmbeddings;
        newEmbeddings.reserve((output.numPatches + 1) * config.hiddenSize);
        
        // Insertar CLS token
        newEmbeddings.insert(newEmbeddings.end(), clsToken.begin(), clsToken.end());
        // Insertar embeddings existentes
        newEmbeddings.insert(newEmbeddings.end(), 
                            output.embeddings.begin(), output.embeddings.end());
        
        // Actualizar máscara (CLS token siempre es válido)
        vector<float> newMask;
        newMask.reserve(output.numPatches + 1);
        newMask.push_back(1.0f);  // CLS token
        newMask.insert(newMask.end(), output.patchMask.begin(), output.patchMask.end());
        
        output.embeddings = newEmbeddings;
        output.patchMask = newMask;
        output.numPatches += 1;
    }
};

#endif