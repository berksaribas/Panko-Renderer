#pragma once
#include <initializer_list>
#include <vector>

//https://twitter.com/SebAaltonen/status/1535253315654762501
template<typename T> struct Slice {
    Slice(std::initializer_list<T> init);
    Slice(T* _data, size_t _size);
    Slice(std::vector<T>& list);
    template<size_t N> Slice(T(&array)[N]);
    explicit Slice(T* _data);
    explicit Slice(T& _data);

    T* m_data;
    size_t m_size;

    T& operator[] (const int i) const;

    const size_t size() const;
    const size_t byte_size() const;
};

template<typename T>
inline Slice<T>::Slice(std::initializer_list<T> init)
{
    m_data = (T*)init.begin();
    m_size = init.size();
}

template<typename T>
inline Slice<T>::Slice(T* _data, size_t _size)
{
    m_data = _data;
    m_size = _size;
}

template<typename T>
inline Slice<T>::Slice(std::vector<T>& list)
{
    m_data = list.data();
    m_size = list.size();
}

template<typename T>
inline Slice<T>::Slice(T* _data)
{
    m_data = _data;
    m_size = 1;
}

template<typename T>
inline Slice<T>::Slice(T& _data)
{
    m_data = &_data;
    m_size = 1;
}

template<typename T>
inline T& Slice<T>::operator[](const int i) const
{
    return m_data[i];
}

template<typename T>
inline const size_t Slice<T>::size() const
{
    return m_size;
}

template<typename T>
inline const size_t Slice<T>::byte_size() const
{
    return m_size * sizeof(T);
}


template<typename T>
template<size_t N>
inline Slice<T>::Slice(T(&array)[N])
{
    m_data = array;
    m_size = sizeof(array) / sizeof(T);
}