#include "frame_allocator.h"
#include <cstdlib>

FrameAllocator::~FrameAllocator()
{
	free(memory);
}

void FrameAllocator::initialize(size_t size)
{
	memory = malloc(size);
	cursor = 0;
	m_size = size;
}

void FrameAllocator::reset()
{
	cursor = 0;
}

void* FrameAllocator::allocate(size_t size, size_t alignment)
{
	size_t remainder = alignment - (cursor % alignment);
	cursor += remainder;

	void* returnPtr = (char*)memory + cursor;
	cursor += size;

	return returnPtr;
}