#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { uint64_t offset; unsigned char before[4], after[4]; } Patch;
_Static_assert(sizeof(Patch)==16, "patch record ABI");
int main(int argc,char **argv){
 if(argc!=3){fprintf(stderr,"usage: apply new-sqrt-eps.tflite patches.bin\n");return 2;}
 const char *prefix="/data/local/tmp/jetlink-litert-20261109/";
 size_t len=strlen(argv[1]);const char *suffix="-sqrt-eps.tflite";size_t slen=strlen(suffix);
 if(strncmp(argv[1],prefix,strlen(prefix))||strstr(argv[1],"..")||len<slen||strcmp(argv[1]+len-slen,suffix))return 2;
 FILE *p=fopen(argv[2],"rb");if(!p)return 2;
 unsigned char magic[4];uint32_t count;uint64_t expected;
 if(fread(magic,1,4,p)!=4||memcmp(magic,"JLP1",4)||fread(&count,4,1,p)!=1||fread(&expected,8,1,p)!=1||!count||count>10000){fclose(p);return 2;}
 Patch *patches=calloc(count,sizeof(Patch));if(!patches){fclose(p);return 2;}
 if(fread(patches,sizeof(Patch),count,p)!=count||fgetc(p)!=EOF){fclose(p);free(patches);return 2;}fclose(p);
 int fd=open(argv[1],O_RDWR);struct stat st;
 if(fd<0||fstat(fd,&st)||st.st_size<0||(uint64_t)st.st_size!=expected){if(fd>=0)close(fd);free(patches);return 3;}
 for(uint32_t i=0;i<count;i++){
  unsigned char old[4];
  if(patches[i].offset>expected-4||pread(fd,old,4,(off_t)patches[i].offset)!=4||memcmp(old,patches[i].before,4)){
   fprintf(stderr,"original bytes mismatch at record %u; no patches applied\n",i);close(fd);free(patches);return 4;
  }
 }
 for(uint32_t i=0;i<count;i++)if(pwrite(fd,patches[i].after,4,(off_t)patches[i].offset)!=4){close(fd);free(patches);return 5;}
 int rc=fsync(fd);close(fd);free(patches);if(rc)return 5;
 printf("applied %u verified four-byte edits to a %llu-byte candidate copy\n",count,(unsigned long long)expected);return 0;
}
