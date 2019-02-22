#include <stdlib.h>
#include <stdio.h>

typedef struct xyzFormat
{
    double x;
    double y;
    double z;
} XYZ;

typedef struct header
{
    size_t numeroDeVariables; //X Y Z RX RY RZ
    // char FormatName;
    unsigned long dataSize;
} FileHeader;

int main(int argc, char **argv)
{
    printf("Tamano Struct Header:%zu\n", sizeof(FileHeader) );
    printf("Tamano Struct XYZ:%zu\n", sizeof(XYZ) );

    //Escribir Archivo en binario, primero se abre el stream
    FILE *write_ptr;
    write_ptr = fopen("test.bin","wb"); //WB=Write Binary

    //Creamos struct con Valores
    XYZ vector[3];
    vector[0].x=0.29;
    vector[0].y=0.19;
    vector[0].z=0.26;
    vector[1].x=0.38;
    vector[1].y=0.61;
    vector[1].z=0.54;
    vector[2].x=0.75;
    vector[2].y=0.84;
    vector[2].z=0.18;

    //Escribimos primero el header del archivo
    FileHeader header;
    header.sizeOfStruct=sizeof(XYZ);
    header.dataSize=3;
    fwrite( &header, sizeof(FileHeader), 1, write_ptr); //Apuntador de los datos, tamano de la estructura,
                                                        //elementos de la estructura, stream donde se escribiraa
    fwrite( (vector), sizeof(XYZ), 3, write_ptr);

    int i;
    /*
    for( i=0; i<header.dataSize; ++i )
    {
        // fwrite ( const void * ptr, size_t size, size_t count, FILE * stream );
        // vector <=> &vector[0]
        // vector+1 <=> &vector[1]
        fwrite( (vector+i), sizeof(XYZ), 1, write_ptr);
    }*/
    fclose(write_ptr);


    //Leer archivo en binario
    FILE *ptr;
    ptr = fopen("test.bin","rb");  // r for read, b for binary

    //Reservar memoria para el header de entrada, nos indicara como leer los datos
    FileHeader headerEntrada;

    //Leer header
    // fread ( void * ptr, size_t size, size_t count, FILE * stream );
    fread(&headerEntrada,sizeof(FileHeader),1,ptr);

    //Iterar el data chunk para conseguir los valores XYZ
    for( i=0; i<headerEntrada.dataSize; ++i )
    {
        XYZ tmp;
        fread(&tmp,sizeof(XYZ),1,ptr);
        printf("Particle[%d].x=%f\n",i,tmp.x);
        printf("Particle[%d].y=%f\n",i,tmp.y);
        printf("Particle[%d].z=%f\n",i,tmp.z);
    }
    fclose(ptr);
    return 0;
}