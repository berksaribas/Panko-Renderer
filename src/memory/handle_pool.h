#pragma once

#include <vector>
#include "handle.h"

template<typename T> class Pool {
public:
	Pool();
	T* get(Handle<T> handle);
	Handle<T> put(T&& data);;
	void remove(Handle<T> handle);
	size_t max_size();
	Handle<T> operator[] (const int i) const;
private:
	std::vector<T> dataList;
	std::vector<uint32_t> generationList;
	std::vector<size_t> freeList;
	std::vector<bool> emptyList;
};

template<typename T>
inline Pool<T>::Pool()
{
	dataList.reserve(256);
	generationList.reserve(256);
	freeList.reserve(256);
	emptyList.reserve(256);
}

template<typename T>
inline T* Pool<T>::get(Handle<T> handle)
{
	if (handle.m_index >= generationList.size()) {
		return nullptr;
	}

	if (handle.m_generation == generationList[handle.m_index]) {
		return &dataList[handle.m_index];
	}

	return nullptr;
}

template<typename T>
inline Handle<T> Pool<T>::put(T&& data)
{
	if (freeList.size() > 0) {
		size_t targetIndex = freeList.back();
		freeList.pop_back();
		dataList[targetIndex] = data;
		emptyList[targetIndex] = false;

		return Handle<T>(targetIndex, generationList[targetIndex]);
	}
	else {
		dataList.push_back(data);
		generationList.push_back(1);
		emptyList.push_back(false);

		return Handle<T>(dataList.size() - 1, 1);
	}
}

template<typename T>
inline void Pool<T>::remove(Handle<T> handle)
{
	if (handle.m_generation == generationList[handle.m_index]) {
		freeList.push_back(handle.m_index);
		generationList[handle.m_index]++;
		emptyList[handle.m_index] = true;
	}
}

template<typename T>
inline size_t Pool<T>::max_size()
{
	return dataList.size();
}

template<typename T>
inline Handle<T> Pool<T>::operator[](const int i) const
{
	if (emptyList[i]) {
		return Handle<T>(0, 0);
	}

	return Handle<T>(i, generationList[i]);
}
