// Ulohy:
//
// 1) Implementace lifu pomoci texturovaci pameti [2 body]
//    - vyzaduje pouziti normalizovanych souradnic pro pristup do textury !!!
//    - inicializace reference - funkce initializeCUDA()
//    viz kernel lifeKernel3()
// 2) Hledani a spocitani stalych tvaru (still lifes) v simulaci [2 body]
//  - nalezeni a spocitani stalych stavu (block, hive, loaf a boat)
//    viz http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
// 3) Hledani oscilatoru v simulaci s periodou 2 [2 body - bonus]
//   - tvary pulsar, toad a beacon


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glut.h>
//#include <GL/gl.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

//#define LIFE_GLOBAL_MEMORY
//#define LIFE_SHARED_MEMORY
#define LIFE_TEXTURE_MEMORY

// odkomentujte pro pouzivani operace modulo pri adresaci globalni pameti
// pri zakomentovani je modulo nahrazeno podminkami
// #define GLOBAL_MEMORY_USE_MODULO

// rozliseni bitmapy a simulacni mrizky
#define IMAGE_WIDTH  768 // 512
#define IMAGE_HEIGHT 768 // 512

// velikost bloku simulacni mrizky
#define BLOCK_DIMENSION 16
// velikost bloku ukladaneho do sdilene pameti
#define SHARED_MEM_BLOCK_DIMENSION (BLOCK_DIMENSION+2)

#define BYTE unsigned char

// tabulka novych stavu (9 pro zive + 9 pro mrtve)
__constant__ BYTE newState[18];

#ifdef LIFE_TEXTURE_MEMORY
cudaArray *cuArray;
texture<BYTE, 2, cudaReadModeElementType> texRef;
#endif

// pocet zivych bunek v aktualni generaci
int *devLivingCells = NULL;
int livingCells;

// struktura bitmapy
typedef struct _Bitmap {
    int width;			// sirka bitmapy v pixelech
    int height;			// vyska bitmapy v pixelech
    uchar4 *pixels;		// ukazatel na bitmapu na strane CPU
    uchar4 *deviceData;	// ukazatel na data bitmapy na GPU
} SBitmap;

SBitmap *bitmap = NULL;				// bitmapa
__device__ BYTE* life1 = NULL;		// simulacni mrizky
__device__ BYTE* life2 = NULL;

BYTE *cpuLife1 = NULL;	// simulacni mrizky pro cpu
BYTE *cpuLife2 = NULL;
BYTE *tmpLife = NULL;
int cpuLivingCells;		// pocet zivych bunek v cpu verzi lifu

BYTE tmpNewState[] = {
        0, 0, 0, 1, 0, 0, 0, 0, 0,    // nove stavy pro mrtve bunky state=0
        0, 0, 1, 1, 0, 0, 0, 0, 0     // nove stavy pro zive bunky state=1
};

// udalosti pro mereni casu v CUDA
cudaEvent_t start, stop;

// grid and block dimensions
dim3 blocks(IMAGE_WIDTH/BLOCK_DIMENSION, IMAGE_HEIGHT/BLOCK_DIMENSION);
dim3 threads(BLOCK_DIMENSION, BLOCK_DIMENSION);

// ukazatel na funkci pro spusteni spravne verze kernelu
void (*lifeKernel)(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height);

// funkce pro osetreni chyb
static void HandleError( cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        scanf(" ");
        exit( EXIT_FAILURE );
    }
}

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )


// vykresleni bitmapy v OpenGL
void DrawImage( void ) {
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

    if(bitmap != NULL) {
        if(bitmap->pixels != NULL)
            glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
    }

    glutSwapBuffers();
}

/* funkce zajistujici aktualizaci simulace - verze pro CPU
 *  in - vstupni simulacni mrizka
 *  out - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */

void lifeKernelCPU(BYTE* in, BYTE* out, int width, int height) {

    cpuLivingCells = 0;

    for(int row=0;row<height;row++) {
        for(int col=0; col<width; col++) {

            int rowAddr = row * width;
            int threadID = rowAddr + col;	// index vlakna
            // indexy sousednich prvku
            int colLeft = (col-1+width) % width;
            int colRight = (col+1) % width;
            int rowTopAddr = ((row+1) % height) * width;
            int rowBottomAddr = ((row-1+height) % height) * width;

            BYTE neighbours;

            // spocitani zivych sousedu
            neighbours = in[rowTopAddr + colLeft];
            neighbours += in[rowTopAddr + col];
            neighbours += in[rowTopAddr + colRight];
            neighbours += in[rowAddr + colLeft];
            neighbours += in[rowAddr + colRight];
            neighbours += in[rowBottomAddr + colLeft];
            neighbours += in[rowBottomAddr + col];
            neighbours += in[rowBottomAddr + colRight];

            // vypocet nove hodnoty dle tabulky ulozene v konstantni pameti
            BYTE oldValue = in[threadID];
            BYTE newValue = tmpNewState[neighbours + 9*oldValue];

            // ulozeni noveho stavu bunky
            out[threadID] = newValue;

            if(newValue == 1)
                cpuLivingCells++;
        }
    }

    std::cout << "Living cells count (CPU): " << cpuLivingCells << std::endl;
}

// funkce pro zapis barvy pixelu do bitmapy, nova barva je odvozena ze stavu simulace
inline __device__ void stateToColor(BYTE oldValue, BYTE newValue, uchar4* bitmap, int bitmapId) {
    uchar4 color;

    color.x = (newValue==0 && oldValue==1) ? 255 : 0;
    color.y = (newValue==1 && oldValue==0) ? 255 : 0;
    color.z = (newValue==1 && oldValue==1) ? 255 : 0;
    color.w = 0;

    bitmap[bitmapId] = color;
}

#ifdef LIFE_GLOBAL_MEMORY

/* kernel zajistujici aktualizaci simulace - verze s globalni pameti
 *  bitmap - bitmapa, ktera se meni dle vzniku a zaniku bunek
 *  in - vstupni simulacni mrizka
 *  out - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
__global__ void lifeKernel1(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height){
 // index (radek, sloupec) zpracovavaneho elementu
 int col = blockIdx.x*blockDim.x + threadIdx.x;
 int row = blockIdx.y*blockDim.y + threadIdx.y;
 int rowAddr = row * width;
 int threadID = rowAddr + col;	// index vlakna

  if(threadID < width*height) {

    // indexy sousednich prvku
#ifdef GLOBAL_MEMORY_USE_MODULO
    int colLeft = (col-1+width) % width;
    int colRight = (col+1) % width;
    int rowTopAddr = ((row+1) % height) * width;
    int rowBottomAddr = ((row-1+height) % height) * width;
#else
    // pozor: modulo je opravdu draha operace, proto je nahrazeno podminkou
    int colLeft = col - 1;
    colLeft = (colLeft < 0) ? (colLeft+width) : colLeft;
    int colRight = col + 1;
    colRight = (colRight > width-1) ? (colRight-width) : colRight;
    int rowTopAddr = row + 1;
    rowTopAddr = width * ((rowTopAddr > height-1) ? (rowTopAddr-height) : rowTopAddr);
    int rowBottomAddr = row - 1;
    rowBottomAddr = width * ((rowBottomAddr < 0) ? (rowBottomAddr+height) : rowBottomAddr);
#endif // GLOBAL_MEMORY_USE_MODULO

    int neighbours;

    // spocitani zivych sousedu
    neighbours = in[rowTopAddr + colLeft];
    neighbours += in[rowTopAddr + col];
    neighbours += in[rowTopAddr + colRight];
    neighbours += in[rowAddr + colLeft];
    neighbours += in[rowAddr + colRight];
    neighbours += in[rowBottomAddr + colLeft];
    neighbours += in[rowBottomAddr + col];
    neighbours += in[rowBottomAddr + colRight];

    // vypocet nove hodnoty dle tabulky ulozene v konstantni pameti
    BYTE oldValue = in[threadID];
    BYTE newValue = newState[neighbours + 9*oldValue];

    // ulozeni noveho stavu bunky
    out[threadID] = newValue;

    // nastaveni barvy pixelu v bitmape dle oldValue a newValue
    stateToColor(oldValue, newValue, bitmap, threadID);
  }
}

// funkce pro spusteni kernelu vyuzivajiciho pouze globalni pamet
void lifeKernelCW1(uchar4* bitmap, BYTE* in,BYTE* out, int width, int height){

  lifeKernel1<<<blocks,threads>>>(bitmap, in, out, width, height);
}

#endif

#ifdef LIFE_SHARED_MEMORY

/* kernel zajistujici aktualizaci simulace - verze se sdilenou pameti
 *  bitmap - bitmapa, ktera se meni dle vzniku a zaniku bunek
 *  in - vstupni simulacni mrizka
 *  out - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
__global__ void lifeKernel2(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height){
  // cache pro ulozeni bloku dat ve sdilene pameti
 __shared__ BYTE sharedData[SHARED_MEM_BLOCK_DIMENSION*SHARED_MEM_BLOCK_DIMENSION];
 int prevBlocksElementsX = blockIdx.x*blockDim.x;
 int prevBlocksElementsY = blockIdx.y*blockDim.y;
 // index (radek, sloupec) zpracovavaneho elementu
 int col = prevBlocksElementsX + threadIdx.x;
 int row = prevBlocksElementsY + threadIdx.y;
 int threadID = row * width + col;

  if(threadID >= width*height)
    return;

  // index elementu v ramci bloku vlaken BLOCK_DIMENSION x BLOCK_DIMENSION po premapovani na 1D pole
  int elementIndex = threadIdx.y*BLOCK_DIMENSION + threadIdx.x;
  // prepocet indexu y bloku vlaken do bloku cache o velikosti SHARED_MEM_BLOCK_DIMENSIONxSHARED_MEM_BLOCK_DIMENSION
  row = elementIndex / SHARED_MEM_BLOCK_DIMENSION;
  //col = elementIndex % SHARED_MEM_BLOCK_DIMENSION;
  col = elementIndex - row*SHARED_MEM_BLOCK_DIMENSION;
  // vypocet indexu (radek, sloupec) odpovidajiciho elementu v globalni pameti
  row = (row - 1 + prevBlocksElementsY + height) % height;
  col = (col - 1 + prevBlocksElementsX + width) % width;
  // index elementu v globalni pameti po premapovani na 1D pole
  int globalIndex = row * width + col;
  // nacteni dat do sdilene pameti -> prvni cislo
  sharedData[elementIndex] = in[globalIndex];

  // posun na dalsi element - na jeden pruchod blok nacte BLOCK_DIMENSION x BLOCK_DIMENSION
  elementIndex += BLOCK_DIMENSION*BLOCK_DIMENSION;

  // test zda druhy nacitany prvek lezi v bloku o velikosti (BLOCK_DIMENSION+2)*(BLOCK_DIMENSION+2)
  // -> nektera vlakna nic nenactou, neni tolik dat
  if( elementIndex < SHARED_MEM_BLOCK_DIMENSION*SHARED_MEM_BLOCK_DIMENSION) {

    row = elementIndex / SHARED_MEM_BLOCK_DIMENSION;
    //col = elementIndex % SHARED_MEM_BLOCK_DIMENSION;
    col = elementIndex - row*SHARED_MEM_BLOCK_DIMENSION;

    row = (row - 1 + prevBlocksElementsY + height) % height;
    col = (col - 1 + prevBlocksElementsX + width) % width;

    globalIndex = row * width + col;
    // nacteni dat do sdilene pameti -> druhe cislo
    sharedData[elementIndex] = in[globalIndex];
  }
  // synchronizace - zajisti ulozeni vsech hodnot do cache pred dalsim pokracovanim
  __syncthreads();

  // index bunky v ramci cache -> pole sharedData
  row = threadIdx.y + 1;
  col = threadIdx.x + 1;

  // predpocitani hodnot vyuzitych pri pristupu k sousednim bunkam
  int rowAddr = row*SHARED_MEM_BLOCK_DIMENSION + col;
  int rowTopAddr = rowAddr + SHARED_MEM_BLOCK_DIMENSION; // (row+1) * SHARED_MEM_BLOCK_DIMENSION;
  int rowBottomAddr = rowAddr - SHARED_MEM_BLOCK_DIMENSION; // (row-1) * SHARED_MEM_BLOCK_DIMENSION;

  BYTE neighbours;

  // spocitani zivych sousedu
  neighbours = sharedData[rowTopAddr - 1];
  neighbours += sharedData[rowTopAddr];
  neighbours += sharedData[rowTopAddr + 1];
  neighbours += sharedData[rowAddr - 1];
  neighbours += sharedData[rowAddr + 1];
  neighbours += sharedData[rowBottomAddr - 1];
  neighbours += sharedData[rowBottomAddr];
  neighbours += sharedData[rowBottomAddr + 1];

  // vypocet nove hodnoty dle tabulky ulozene v konstantni pameti
  BYTE oldValue = sharedData[rowAddr];
  BYTE newValue = newState[neighbours + 9*oldValue];

  // ulozeni noveho stavu bunky
  out[threadID] = newValue;

  // nastaveni barvy pixelu v bitmape dle oldValue a newValue
  stateToColor(oldValue, newValue, bitmap, threadID);
}

// funkce pro spusteni kernelu se sdilenou pameti
void lifeKernelCW2(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height){

  lifeKernel2<<<blocks,threads>>>( bitmap, in, out, width, height);
}

#endif

#ifdef LIFE_TEXTURE_MEMORY

/* kernel zajistujici aktualizaci simulace - verze s pameti textur
 *  bitmap - bitmapa, ktera se meni dle vzniku a zaniku bunek
 *  in - vstupni simulacni mrizka
 *  out - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
__global__ void lifeKernel3(uchar4* bitmap, BYTE* out, int width, int height){

    // DOPLNTE !!!
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int rowAddr = row * width;
    int threadID = rowAddr + col;	// index vlakna

    float coord_per_block_horizontal = (float) width / BLOCK_DIMENSION;
    float coord_per_block_vertical = (float) height / BLOCK_DIMENSION;

    int neighbors = 0;

    if(threadID < width*height) {
        neighbors += tex2D(texRef, 0.1, 0.1);

    }
}

// funkce pro spusteni kernelu se sdilenou pameti
void lifeKernelCW3(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height){

    lifeKernel3<<<blocks,threads>>>(bitmap, out, width, height);

    // kopirovani vystupniho pole do cuda array, ktere je spojene s referenci na texturu
    cudaMemcpyToArray(cuArray, 0, 0, out, width*height*sizeof(BYTE), cudaMemcpyDeviceToDevice);
}

#endif

/* kernel pocitajici pocet zivych bunek, vyuziva sdilenou pamet
 *  in - vstupni simulacni mrizka
 *  livingCellsCount - pocet zivych bunek ve vstupni simulacni mrizce
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
__global__ void countCells(BYTE* in, int *livingCellsCount, int width, int height){
    // pole ve sdilene pameti pro ulozeni hodnot budek a nasledny vypocet paralelni redukce
    extern __shared__ int cache[];

    // index (radek, sloupec) zpracovavaneho elementu
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int threadID = row * width + col;	// index vlakna
    int cacheIndex = threadIdx.y*blockDim.x + threadIdx.x;

    // nacteni hodnoty do sdilene pameti
    cache[cacheIndex] = in[threadID];

    // synchronizace vl\E1ken v r\E1mci bloku \96 zajist\EDm si dokon\E8en\ED v\9Aech z\E1pis\F9 do cache
    __syncthreads();

    // pro paralelni redukci implementovanou v nasledujicim kodu
    // musi byt pocet vlaken v bloku mocnina 2
    // pocet kroku redukce je roven dvojkovemu logaritmu poctu vlaken v bloku
    int step = blockDim.x*blockDim.y / 2;

    while( step != 0 ) {
        if(cacheIndex < step)
            cache[cacheIndex] += cache[cacheIndex + step];

        // synchronizace vlaken po provedeni kazde faze redukce
        __syncthreads();
        // zmenseni kroku pro dalsi fazi redukce
        step /= 2;
    }

    // zapis vysledku do vystupniho pole
    // zapis provadi pouze prvni vlakno
    if(cacheIndex == 0)
        atomicAdd(livingCellsCount, cache[0]);
}


// funkce pro spusteni kernelu + priprava potrebnych dat a struktur
void callKernelCUDA(void) {

    // ulozeni pocatecniho casu
    CHECK_ERROR( cudaEventRecord( start, 0 ) );

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    lifeKernel(bitmap->deviceData, life1, life2, bitmap->width, bitmap->height);

#ifndef LIFE_TEXTURE_MEMORY
    // prohozeni ukazatelu (u textur pouzit pouze life2)
  swap(life1, life2);
#endif

    // ulozeni casu ukonceni simulace
    CHECK_ERROR( cudaEventRecord( stop, 0 ) );
    CHECK_ERROR( cudaEventSynchronize( stop ) );

    float elapsedTime;

    // vypis casu simulace
    CHECK_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    // kopirovani bitmapy zpet na CPU pro zobrazeni
    CHECK_ERROR( cudaMemcpy( bitmap->pixels, bitmap->deviceData, bitmap->width*bitmap->height*sizeof(uchar4), cudaMemcpyDeviceToHost ) );

    // spocitani zivych bunek
    CHECK_ERROR( cudaMemset( devLivingCells, 0, sizeof(int) ) );

#ifndef LIFE_TEXTURE_MEMORY
    countCells<<<blocks,threads,threads.x*threads.y*sizeof(int)>>>(life1, devLivingCells, bitmap->width, bitmap->height);
#else
    countCells<<<blocks,threads,threads.x*threads.y*sizeof(int)>>>(life2, devLivingCells, bitmap->width, bitmap->height);
#endif

    cudaMemcpy( &livingCells, devLivingCells, sizeof(int), cudaMemcpyDeviceToHost );

    std::cout << "Living cells count (GPU): " << livingCells << std::endl;

    // krok simulace life game na CPU
    lifeKernelCPU(cpuLife1, cpuLife2, bitmap->width, bitmap->height);
    swap(cpuLife1, cpuLife2);

#ifndef LIFE_TEXTURE_MEMORY
    cudaMemcpy( tmpLife, life1, bitmap->width*bitmap->height*sizeof(BYTE), cudaMemcpyDeviceToHost );
#else
    cudaMemcpy( tmpLife, life2, bitmap->width*bitmap->height*sizeof(BYTE), cudaMemcpyDeviceToHost );
#endif

    int diffs = 0;

    // porovnani vysledku CPU simulace a GPU simulace
    for(int row=0;row<bitmap->height;row++) {
        for(int col=0; col<bitmap->width; col++) {

            int rowAddr = row * bitmap->width;
            int threadID = rowAddr + col;	// index vlakna

            if(cpuLife1[threadID] != tmpLife[threadID])
                diffs++;
        }
    }

    if(diffs != 0)
        std::cout << "ERROR: " << diffs << " differences between CPU & GPU grid" << std::endl;

    if(cpuLivingCells != livingCells)
        std::cout << "ERROR: incorrect cells count (GPU) !" << std::endl;
}

// funkce je volana opakovane -> vytvoreni animace
void idleFunc() {
    // zavolani kernelu pro aktualizaci simulace
    callKernelCUDA();
    // prekresleni
    glutPostRedisplay();
}

// inicializace CUDA - alokace potrebnych dat a vygenerovani pocatecniho stavu lifu
void initializeCUDA(void) {

    // alokace struktury bitmapy
    bitmap = (SBitmap *)malloc(sizeof(bitmap));
    bitmap->width = IMAGE_WIDTH;
    bitmap->height = IMAGE_HEIGHT;

    cudaHostAlloc((void**)&(bitmap->pixels), bitmap->width*bitmap->height*sizeof(uchar4), cudaHostAllocDefault);
    //bitmap->pixels = new uchar4[bitmap->width*bitmap->height];

    // alokovani mista pro bitmapu na GPU
    int bitmapSize = bitmap->width*bitmap->height;
    CHECK_ERROR( cudaMalloc( (void**)&(bitmap->deviceData), bitmapSize*sizeof(uchar4) ) );
#ifndef LIFE_TEXTURE_MEMORY
    CHECK_ERROR( cudaMalloc( (void**)&(life1), bitmapSize*sizeof(BYTE) ) );
#endif
    CHECK_ERROR( cudaMalloc( (void**)&(life2), bitmapSize*sizeof(BYTE) ) );

    CHECK_ERROR( cudaMalloc( (void**)&(devLivingCells), sizeof(int) ) );

    cpuLife1 = (BYTE *)malloc(bitmapSize*sizeof(BYTE));
    cpuLife2 = (BYTE *)malloc(bitmapSize*sizeof(BYTE));
    tmpLife = (BYTE *)malloc(bitmapSize*sizeof(BYTE));

    //srand(time(NULL));
    srand(0);

    // inicializace pocatecniho stavu lifu
    for (int i=0; i<bitmapSize; i++) {
        cpuLife1[i] = (BYTE)(rand() % 2);
    }

#ifndef LIFE_TEXTURE_MEMORY
    // prekopirovani pocatecniho stavu do GPU
  cudaMemcpy( life1, cpuLife1, bitmapSize*sizeof(BYTE), cudaMemcpyHostToDevice );
#else
    // format alokovaneho pole \96 kazdy texel bude reprezentovan jednim 8-bitovym unsigned charem
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    // alokace pole na GPU
    cudaMallocArray(&cuArray, &channelDesc, bitmap->width, bitmap->height);
    // naplneni pole na GPU, tj. kopirovani dat z adresy hostData na GPU
    cudaMemcpyToArray(cuArray, 0, 0, cpuLife1, bitmapSize*sizeof(BYTE), cudaMemcpyHostToDevice);

    // DOPLNTE !!!!
    // nastaveni parametru textury \96 zpusob filtrovani, zpracovani hodnot mimo rozsah, \85
    // texRef.addressMode ...
    texRef.normalized = true;
    texRef.filterMode = cudaFilterModePoint;
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;

    // svazani reference na texturu se skutecnymi daty
    cudaBindTextureToArray(texRef, cuArray, channelDesc);
#endif

    // nakopirovani tabulky novych stavu do konstantni pameti
    cudaMemcpyToSymbol( newState, tmpNewState, sizeof(BYTE) * 18) ;

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR( cudaEventCreate( &start ) );
    CHECK_ERROR( cudaEventCreate( &stop ) );
}

// funkce volana pri ukonceni aplikace, uvolni vsechy prostredky alokovane v CUDA
void finalizeCUDA(void) {

#ifdef LIFE_TEXTURE_MEMORY
    // uvoln\ECn\ED alokovan\E9ho pole na GPU
    cudaUnbindTexture(texRef);
    cudaFreeArray(cuArray);
#endif

    // uvolneni bitmapy - na CPU i GPU
    if(bitmap != NULL) {
        if(bitmap->pixels != NULL) {
            // uvolneni bitmapy na CPU
            cudaFreeHost(bitmap->pixels);
            bitmap->pixels = NULL;
        }
        if(bitmap->deviceData != NULL) {
            // uvolneni bitmapy na GPU
            cudaFree(bitmap->deviceData);
            bitmap->deviceData = NULL;
        }
        free(bitmap);
    }

    // uvolneni simulacnich mrizek pro CPU variantu lifu
    free(cpuLife1);
    free(cpuLife2);
    free(tmpLife);

    // zruseni struktur udalosti
    CHECK_ERROR( cudaEventDestroy( start ) );
    CHECK_ERROR( cudaEventDestroy( stop ) );
}

// zpracovani udalosti klavesnice
static void HandleKeys(unsigned char key, int x, int y) {
    switch (key) {
        case 27:	// ESC
            finalizeCUDA();
            exit(0);
    }
}


int main(int argc, char **argv) {

#ifdef LIFE_GLOBAL_MEMORY
    lifeKernel = lifeKernelCW1;
#endif
#ifdef LIFE_SHARED_MEMORY
    lifeKernel = lifeKernelCW2;
#endif
#ifdef LIFE_TEXTURE_MEMORY
    lifeKernel = lifeKernelCW3;
#endif

    initializeCUDA();

    glutInit(&argc, argv);

    glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );

    glutCreateWindow("Life Game");
    glutDisplayFunc(DrawImage);
    glutKeyboardFunc(HandleKeys);

    glutIdleFunc(idleFunc);

    glutMainLoop();

    return 0;
}