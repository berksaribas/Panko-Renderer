#pragma once

#include "vk_rendergraph_types.h"
#include <span>

namespace Vrg
{
inline void hash_combine(std::size_t& seed)
{
}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

struct DescriptorSetCache
{
    std::vector<DescriptorSet> descriptorSets;
    bool operator==(DescriptorSetCache const& rhs) const noexcept
    {
        if (descriptorSets.size() == rhs.descriptorSets.size())
        {
            for (int i = 0; i < descriptorSets.size(); i++)
            {
                if (descriptorSets[i].type != rhs.descriptorSets[i].type ||
                    descriptorSets[i].imageLayout != rhs.descriptorSets[i].imageLayout ||
                    descriptorSets[i].imageView != rhs.descriptorSets[i].imageView ||
                    descriptorSets[i].buffer != rhs.descriptorSets[i].buffer ||
                    descriptorSets[i].image != rhs.descriptorSets[i].image)
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }
        return true;
    }
};

struct DescriptorSetLayoutCache
{
    std::vector<VkDescriptorType> descriptorTypes;
    bool operator==(DescriptorSetLayoutCache const& rhs) const noexcept
    {
        if (descriptorTypes.size() == rhs.descriptorTypes.size())
        {
            return memcmp(descriptorTypes.data(), rhs.descriptorTypes.data(),
                          sizeof(VkDescriptorType) * descriptorTypes.size()) == 0;
        }
        return false;
    }
};

struct PipelineLayoutCache
{
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkPushConstantRange> pushRanges;
    bool operator==(PipelineLayoutCache const& rhs) const noexcept
    {
        if (descriptorSetLayouts.size() == rhs.descriptorSetLayouts.size())
        {
            for (int i = 0; i < descriptorSetLayouts.size(); i++)
            {
                if (descriptorSetLayouts[i] != rhs.descriptorSetLayouts[i])
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }

        if (pushRanges.size() == rhs.pushRanges.size())
        {
            for (int i = 0; i < pushRanges.size(); i++)
            {
                if (pushRanges[i].offset != rhs.pushRanges[i].offset ||
                    pushRanges[i].size != rhs.pushRanges[i].size ||
                    pushRanges[i].stageFlags != rhs.pushRanges[i].stageFlags)
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }

        return true;
    }
};

struct ImageViewCache
{
    VkImage image;
    ImageView imageView;

    bool operator==(ImageViewCache const& rhs) const noexcept
    {
        return image == rhs.image && imageView == rhs.imageView;
    }
};

struct ImageMipCache
{
    VkImage image;
    uint32_t mip;

    bool operator==(ImageMipCache const& rhs) const noexcept
    {
        return image == rhs.image && mip == rhs.mip;
    }
};

struct DescriptorSetCache_hash
{
    size_t operator()(DescriptorSetCache const& x) const noexcept
    {
        size_t h = 0;
        for (int i = 0; i < x.descriptorSets.size(); i++)
        {
            hash_combine(h, x.descriptorSets[i].type, x.descriptorSets[i].imageLayout,
                         x.descriptorSets[i].imageView.baseMipLevel,
                         x.descriptorSets[i].imageView.mipLevelCount,
                         x.descriptorSets[i].imageView.sampler,
                         reinterpret_cast<uint64_t>(x.descriptorSets[i].buffer),
                         reinterpret_cast<uint64_t>(x.descriptorSets[i].image));
        }
        return h;
    }
};

struct DescriptorSetLayoutCache_hash
{
    size_t operator()(DescriptorSetLayoutCache const& x) const noexcept
    {
        size_t h = 0;

        for (int i = 0; i < x.descriptorTypes.size(); i++)
        {
            hash_combine(h, x.descriptorTypes[i]);
        }
        return h;
    }
};

struct PipelineLayoutCache_hash
{
    size_t operator()(PipelineLayoutCache const& x) const noexcept
    {
        size_t h = 0;
        for (int i = 0; i < x.descriptorSetLayouts.size(); i++)
        {
            hash_combine(h, reinterpret_cast<uint64_t>(x.descriptorSetLayouts[i]));
        }
        for (int i = 0; i < x.pushRanges.size(); i++)
        {
            hash_combine(h, x.pushRanges[i].offset, x.pushRanges[i].size,
                         x.pushRanges[i].stageFlags);
        }
        return h;
    }
};

struct ImageViewCache_hash
{
    size_t operator()(ImageViewCache const& x) const noexcept
    {
        size_t h = 0;
        hash_combine(h, reinterpret_cast<uint64_t>(x.image), x.imageView.baseMipLevel,
                     x.imageView.mipLevelCount, x.imageView.sampler);
        return h;
    }
};

struct ImageMipCache_hash
{
    size_t operator()(ImageMipCache const& x) const noexcept
    {
        size_t h = 0;
        hash_combine(h, reinterpret_cast<uint64_t>(x.image), x.mip);
        return h;
    }
};
} // namespace Vrg