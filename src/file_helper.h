#ifndef FILE_HELPER_H_
#define FILE_HELPER_H_

#include <fstream>

void save_binary(std::string filename, void* data, size_t size);
size_t seek_file(std::string filename);
void load_binary(std::string filename, void* destination, size_t size);

#ifdef FILE_HELPER_IMPL

void save_binary(std::string filename, void* data, size_t size)
{
    FILE* ptr;
    fopen_s(&ptr, filename.c_str(), "wb");
    fwrite(data, size, 1, ptr);
    fclose(ptr);
}

size_t seek_file(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void load_binary(std::string filename, void* destination, size_t size)
{
    FILE* ptr;
    fopen_s(&ptr, filename.c_str(), "rb");
    fread_s(destination, size, size, 1, ptr);
    fclose(ptr);
}

#endif // FILE_HELPER_IMPL

#endif