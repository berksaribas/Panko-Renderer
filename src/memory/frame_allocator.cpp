#include "frame_allocator.h"
#include <cstdlib>

FrameAllocator::~FrameAllocator()
{
    free(m_memory);
}

void FrameAllocator::initialize(size_t size)
{
    m_memory = malloc(size);
    m_cursor = 0;
    m_size = size;
}

void FrameAllocator::reset()
{
    m_cursor = 0;
}

static size_t calculate_padding(const size_t baseAddress, const size_t alignment)
{
    const size_t multiplier = (baseAddress / alignment) + 1;
    const size_t alignedAddress = multiplier * alignment;
    const size_t padding = alignedAddress - baseAddress;
    return padding;
}

void* FrameAllocator::allocate(size_t size, size_t alignment)
{
    size_t padding = 0;
    size_t paddedAddress = 0;
    const size_t currentAddress = reinterpret_cast<size_t>(m_memory) + m_cursor;

    if (alignment != 0 && m_cursor % alignment != 0)
    {
        // Alignment is required. Find the next aligned memory address and update offset
        padding = calculate_padding(currentAddress, alignment);
    }

    if (m_cursor + padding + size > m_size)
    {
        return nullptr;
    }

    m_cursor += padding;
    const size_t nextAddress = currentAddress + padding;
    m_cursor += size;

    return reinterpret_cast<void*>(nextAddress);
}