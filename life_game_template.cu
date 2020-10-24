// Ulohy:
//
// 1) Implementace kernelu countCells() [2 body]
//    - kernel spocte pocet zivych bunek v dane generaci
//    - 1. krok: pomoci paraleni redukce v ramci bloku spocte pocet zivych bunek (sdilena pamet)
//    - 2. krok: soucty za bloky budou ulozeny do globalni pameti pomoci atomicke operace
//    
// 2) Implementace lifu pomoci sdilene pameti - kernel lifeKernel2()  [2 body]
//    - 1.krok: kazdy blok vlaken (BLOCK_DIMENSION x BLOCK_DIMENSION) nacte do sdilene pameti vsechny prvky, ktere bude
//         potrebovat (vcetne okraje), tj. (BLOCK_DIMENSION+2 x BLOCK_DIMENSION+2) prvku
//         -> kazde vlakno tedy nacte dva prvky (linearizace bloku vlaken -> 1D index -> pouzit pro cteni rozsireneho bloku
//            s okraji)
//    - 2.krok: kazde vlakno spocte novy stav jedne bunky
//    - 3.krok: kazde vlakno ulozi novy stav jedne bunky 
//
//    - na pocatku je spousten kernel resici life v globalni pameti, pro spusteni kernelu vyuzivajiciho sdilenou pamet
//       je potreba odkomentovat #define LIFE_SHARED_MEMORY a zakomentovat #define LIFE_GLOBAL_MEMORY
//
// Vyresene ulohy zaslat emailem se subjectem "B4M39GPU - TASKS - 1", nejdpozdeji do stredy 21.10. 23:59.


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glut.h>

using namespace std;

#define LIFE_GLOBAL_MEMORY
//#define LIFE_SHARED_MEMORY

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
        int colLeft = (col-1+width) % width;
        int colRight = (col+1) % width;
        int rowTopAddr = ((row+1) % height) * width;
        int rowBottomAddr = ((row-1+height) % height) * width;

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

 // DOPLNTE !!!
 // jako vzor pouzijte verzi s globalni pameti - kernel lifeKernel1()

}

// funkce pro spusteni kernelu se sdilenou pameti
void lifeKernelCW2(uchar4* bitmap, BYTE* in, BYTE* out, int width, int height){

  lifeKernel2<<<blocks,threads>>>( bitmap, in, out, width, height);
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

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int rowAddr = row * width;
    int threadID_global = rowAddr + col; 

    int threadID = blockDim.y * threadIdx.y + threadIdx.x;

    unsigned int step = 0;

	// nacteni hodnoty do sdilene pameti
        if (threadID < width*height && in[threadID_global] == 1) {
            cache[threadID] = 1;
        } else {
            cache[threadID] = 0;
        }
        __syncthreads();

	// paralelni reducke
        for (step = 1; step < blockDim.x; step *= 2) {
            if (threadID % (2 * step) == 0) {
                cache[threadID] += cache[threadID + step];
            }

            __syncthreads();
        }

        // zapis provadi pouze prvni vlakno v bloku
        if (threadID == 0) {
            atomicAdd(livingCellsCount, cache[0]);
        }
}


// funkce pro spusteni kernelu + priprava potrebnych dat a struktur
void callKernelCUDA(void) {

    // ulozeni pocatecniho casu
    CHECK_ERROR( cudaEventRecord( start, 0 ) );

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    lifeKernel(bitmap->deviceData, life1, life2, bitmap->width, bitmap->height);

    // prohozeni ukazatelu
    swap(life1, life2);

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

    countCells<<<blocks,threads,threads.x*threads.y*sizeof(int)>>>(life1, devLivingCells, bitmap->width, bitmap->height);

    cudaMemcpy( &livingCells, devLivingCells, sizeof(int), cudaMemcpyDeviceToHost );

    std::cout << "Living cells count (GPU): " << livingCells << std::endl;

    // krok simulace life game na CPU
    lifeKernelCPU(cpuLife1, cpuLife2, bitmap->width, bitmap->height);
    swap(cpuLife1, cpuLife2);

    cudaMemcpy( tmpLife, life1, bitmap->width*bitmap->height*sizeof(BYTE), cudaMemcpyDeviceToHost );

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
    CHECK_ERROR( cudaMalloc( (void**)&(life1), bitmapSize*sizeof(BYTE) ) );
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

    // prekopirovani pocatecniho stavu do GPU
    cudaMemcpy( life1, cpuLife1, bitmapSize*sizeof(BYTE), cudaMemcpyHostToDevice );

    // nakopirovani tabulky novych stavu do konstantni pameti
    cudaMemcpyToSymbol( newState, tmpNewState, sizeof(BYTE) * 18) ;

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR( cudaEventCreate( &start ) );
    CHECK_ERROR( cudaEventCreate( &stop ) );
}

// funkce volana pri ukonceni aplikace, uvolni vsechy prostredky alokovane v CUDA 
void finalizeCUDA(void) {

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
