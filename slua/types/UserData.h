/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <array>

namespace slua
{
	inline const size_t TYPE_ID_SIZE = 2;

// After all typeids are initialized, this will contain a value that will always be more than any existing type id
	inline uint16_t TYPE_ID_COUNT=0;

	template<typename TYPE>
	inline const uint16_t _typeId = TYPE_ID_COUNT++;

	template<typename TYPE>
	inline uint16_t getTypeId() {
		return _typeId<TYPE>;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(PTR_T) to find where to change stuff
	template<typename TYPE,typename PTR_T>
	inline void setTypeIdSeperate(PTR_T* ptr)
	{
		uint8_t* dataPtr = reinterpret_cast<uint8_t*>(ptr);

		const uint16_t typeId = getTypeId<TYPE>();

		dataPtr[sizeof(PTR_T)+0] = typeId & 0xFF;
		dataPtr[sizeof(PTR_T)+1] = typeId >> 8;
	}
	// Changes the last 2 bytes into the types id
	// Uses sizeof(TYPE) to find where to change stuff
	template<typename TYPE>
	inline void setTypeId(TYPE* ptr) {
		setTypeIdSeperate<TYPE,TYPE>(ptr);
	}


	// Checks the last 2 bytes, returns true if all is good
	// Uses sizeof(PTR_T) to find where type id is at
	template<typename TYPE, typename PTR_T>
	inline bool checkTypeIdSeperate(const PTR_T* ptr) {
		const uint8_t* dataPtr = reinterpret_cast<const uint8_t*>(ptr);

		const uint16_t typeId = getTypeId<TYPE>();
		return (
			dataPtr[sizeof(PTR_T) + 0] == (typeId & 0xFF))
			&& (
				dataPtr[sizeof(PTR_T) + 1] == (typeId >> 8));
	}
	// Checks the last 2 bytes, returns true if all is good
	// Uses sizeof(TYPE) to find where type id is at
	template<typename TYPE>
	inline bool checkTypeId(const TYPE* ptr) {
		return checkTypeIdSeperate<TYPE, TYPE>(ptr);
	}
}