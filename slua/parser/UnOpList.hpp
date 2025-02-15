/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <memory>

#include "Enums.hpp"
#include <slua/ErrorType.hpp>

namespace sluaParse
{
	static_assert(sizeof(UnOpType) == 1);
	struct UnOpList
	{
		//TODO: switch to realloc[] once that exists

	private:
		constexpr static size_t SMALL_SIZE = 14;
		constexpr static size_t MAX_LARGE_SIZE = (1ULL << ((sizeof(size_t) - 1) * 8)) - 0xFF;

		union
		{
			struct
			{
				UnOpType first14[SMALL_SIZE] = {
					UnOpType::NONE
				};
				uint8_t size = 0;
				uint8_t isBig = 0;
			} small{};
			struct
			{
				UnOpType* ptr;
				size_t size : ((sizeof(size_t) - 1) * 8);

				size_t reserve : 8; // 0 means not big, 1 means no reserved ops, 2 means one op reserved
			} large;
		};
		constexpr bool m_isSmall() const {
			return small.isBig == 0;
		}
	public:
		constexpr UnOpList() = default;

		constexpr size_t size() const {
			return m_isSmall() ? small.size : large.size;
		}
		constexpr static size_t max_size() {
			return MAX_LARGE_SIZE;
		}
		const UnOpType& at(const size_t idx) const
		{
			if (m_isSmall())
			{
				if (idx >= small.size)
					SLua_panic();
				return small.first14[idx];
			}
			if (idx >= large.size)
				SLua_panic();
			return large.ptr[idx];
		}
		void push_back(const UnOpType t)
		{
			if (m_isSmall())
			{
				if (small.size >= SMALL_SIZE)
				{
					constexpr uint8_t SZ = SMALL_SIZE + 1;
					UnOpType* ptr = (UnOpType*)std::malloc(SZ * 2);

					if (ptr == nullptr)
						throw std::bad_alloc();

					std::copy_n(small.first14, SMALL_SIZE, ptr);
					ptr[SMALL_SIZE] = t;

					large.ptr = ptr;
					large.size = SZ;
					large.reserve = SZ + 1;//+1 for storage method

					return;//Done!
				}
				//no need to convert!

				small.first14[small.size] = t;
				small.size++;

				return;//Done!
			}
			//Its big
			if (large.reserve == 1) // 1, cuz 0 means its not large
			{// (re)Alloc
				const uint8_t reserv = (uint8_t)std::min((size_t)UINT8_MAX, large.size + 1);//+1 for storage method

				if (reserv + large.size > MAX_LARGE_SIZE)
					SLua_panic();

				UnOpType* ptr = (UnOpType*)std::realloc(large.ptr, large.size + reserv - 1);// -1, cuz large.reserve is stored in +1

				if (ptr == nullptr)
					throw std::bad_alloc();

				large.reserve = reserv;
				large.ptr = ptr;
			}

			large.reserve--;

			large.ptr[large.size] = t;
			large.size++;
		}

		struct UnOpListIterator
		{
			using iterator_category = std::forward_iterator_tag;
			using value_type = UnOpType;
			using difference_type = std::ptrdiff_t;
			using pointer = const UnOpType*;
			using reference = const UnOpType&;

			UnOpListIterator(const UnOpList* list, size_t index) : m_list(list), m_index(index) {}

			reference operator*() const {
				return m_list->at(m_index);
			}

			pointer operator->() const {
				return &m_list->at(m_index);
			}

			UnOpListIterator& operator++() {
				++m_index;
				if (m_index > m_list->size())
					SLua_panic();
				return *this;
			}

			UnOpListIterator operator++(int) {
				UnOpListIterator tmp = *this;
				++(*this);
				return tmp;
			}

			bool operator==(const UnOpListIterator& other) const {
				return m_list == other.m_list && m_index == other.m_index;
			}

		private:
			const UnOpList* m_list;
			size_t m_index;
		};

		using iterator = UnOpListIterator;
		using const_iterator = UnOpListIterator;

		iterator begin() const {
			return iterator(this, 0);
		}

		iterator end() const {
			return iterator(this, size());
		}

		~UnOpList()
		{
			if (!m_isSmall())
				std::free(large.ptr);
		}
		UnOpList(UnOpList&& o) noexcept {
			//No need to check if small
			this->operator=<true>(std::move(o));
		}
		template<bool NO_SMALL_CHECK=false>
		UnOpList& operator=(UnOpList&& o) noexcept {
			if constexpr (!NO_SMALL_CHECK)
			{
				if (!m_isSmall())
					std::free(large.ptr);
			}

			if (o.small.isBig == 0)//Other one, NOT THIS one
			{// its small
				std::copy_n(o.small.first14, SMALL_SIZE, small.first14);
				small.size = o.small.size;
				small.isBig = 0;
				return *this;
			}
			//Move over data
			large.size = o.large.size;
			large.ptr = o.large.ptr;
			large.reserve = o.large.reserve;
			//Clean other up
			o.small.isBig = 0;
			o.small.size = 0;
			return *this;
		}
	};
	static_assert(sizeof(UnOpList) == 16);
}