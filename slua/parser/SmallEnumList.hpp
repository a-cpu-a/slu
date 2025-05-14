/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <memory>

#include "Enums.hpp"
#include <slua/ErrorType.hpp>

namespace slua::parse
{
	//TODO: reuse this code more... later

	template<class T>
	struct SmallEnumList
	{
		static_assert(sizeof(T) == 1);

		//TODO: switch to realloc[] once that exists

	private:
		constexpr static size_t SMALL_SIZE = 14;
		constexpr static size_t MAX_LARGE_SIZE = (1ULL << ((sizeof(size_t) - 1) * 8)) - 0xFF;

		union
		{
			struct
			{
				T first14[SMALL_SIZE] = {
					T::NONE
				};
				uint8_t size = 0;
				uint8_t isBig = 0;
			} small{};
			struct
			{
				T* ptr;
				size_t size : ((sizeof(size_t) - 1) * 8);

				size_t reserve : 8; // 0 means not big, 1 means no reserved ops, 2 means one op reserved
			} large;
		};
		constexpr bool m_isSmall() const {
			return small.isBig == 0;
		}
	public:
		constexpr SmallEnumList() = default;

		constexpr size_t size() const {
			return m_isSmall() ? small.size : large.size;
		}
		constexpr static size_t max_size() {
			return MAX_LARGE_SIZE;
		}
		const bool isBack(const T cmp) const
		{
			if (size() == 1)return false;

			return at(size() - 1)==cmp;
		}
		const T& at(const size_t idx) const
		{
			if (m_isSmall())
			{
				Slua_require(idx < small.size);
				return small.first14[idx];
			}
			Slua_require(idx < large.size);
			return large.ptr[idx];
		}
		void erase_back()
		{
			Slua_require(size()!=0);
			if (m_isSmall())
			{
				small.size--;
				return;
			}
			large.size--;

			if (large.reserve < 0xFF)
				large.reserve++;
		}
		void push_back(const T t)
		{
			if (m_isSmall())
			{
				if (small.size >= SMALL_SIZE)
				{
					constexpr uint8_t SZ = SMALL_SIZE + 1;
					T* ptr = (T*)std::malloc(SZ * 2);

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


				Slua_require((reserv + large.size) <= MAX_LARGE_SIZE);

				T* ptr = (T*)std::realloc(large.ptr, large.size + reserv - 1);// -1, cuz large.reserve is stored in +1

				if (ptr == nullptr)
					throw std::bad_alloc();

				large.reserve = reserv;
				large.ptr = ptr;
			}

			large.reserve--;

			large.ptr[large.size] = t;
			large.size++;
		}

		template<bool rev>
		struct SmallEnumListIterator
		{
			using iterator_category = std::forward_iterator_tag;
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using pointer = const T*;
			using reference = const T&;

			SmallEnumListIterator(const SmallEnumList* list, size_t index) : m_list(list), m_index(index) {}

			inline static constexpr uint8_t revOffset = rev ? 1 : 0;

			reference operator*() const {
				return m_list->at(m_index- revOffset);
			}

			pointer operator->() const {
				return &*this;
			}

			SmallEnumListIterator& operator++() {
				if constexpr (rev)
				{
					Slua_require(m_index != 0);
					--m_index;
					return *this;
				}
				++m_index;
				Slua_require(m_index <= m_list->size());//allow == size, as that is end()
				return *this;
			}

			SmallEnumListIterator operator++(int) {
				SmallEnumListIterator tmp = *this;
				++(*this);
				return tmp;
			}

			bool operator==(const SmallEnumListIterator& other) const {
				return m_list == other.m_list && m_index == other.m_index;
			}

		private:
			const SmallEnumList* m_list;
			size_t m_index;
		};

		using iterator = SmallEnumListIterator<false>;
		using const_iterator = SmallEnumListIterator<false>;
		using reverse_iterator = SmallEnumListIterator<true>;
		using const_reverse_iterator = SmallEnumListIterator<true>;

		iterator begin() const {
			return iterator(this, 0);
		}
		const_iterator cbegin() const {
			return const_iterator(this, 0);
		}
		iterator end() const {
			return iterator(this, size());
		}
		const_iterator cend() const {
			return const_iterator(this, size());
		}
		reverse_iterator rbegin() const {
			return reverse_iterator(this, size());
		}
		const_reverse_iterator crbegin() const {
			return const_reverse_iterator(this, size());
		}
		reverse_iterator rend() const {
			return reverse_iterator(this, 0);
		}
		const_reverse_iterator crend() const {
			return const_reverse_iterator(this, 0);
		}

		~SmallEnumList()
		{
			if (!m_isSmall())
				std::free(large.ptr);
		}
		SmallEnumList(SmallEnumList&& o) noexcept {
			//No need to check if small
			this->operator=<true>(std::move(o));
		}
		template<bool NO_SMALL_CHECK=false>
		SmallEnumList& operator=(SmallEnumList&& o) noexcept {
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
	static_assert(sizeof(SmallEnumList<UnOpType>) == 16);
}