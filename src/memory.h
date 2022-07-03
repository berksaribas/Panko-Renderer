#pragma once

class FrameAllocator
{
public:
	~FrameAllocator();
	void initialize(size_t size);
	void reset();
	void* allocate(size_t size, size_t alignment);
	template <typename T>
	T* allocate() {
		return (T*)allocate(sizeof(T), alignof(T));
	}
	template <typename T>
	T* allocate_array(size_t size) {
		return (T*)allocate(sizeof(T) * size, alignof(T));
	}
private:
	void* memory;
	size_t cursor;
	size_t m_size;
};