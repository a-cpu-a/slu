/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <shared_mutex>

#include <slua/parser/State.hpp>
#include <slua/lang/BasicState.hpp>
#include <slua/midlevel/MidState.hpp>

namespace slua::mlvl
{
	struct GenObj
	{
		//Restrictions:
		bool varLike : 1 = false;
		bool typeLike : 1 = false;
		bool useable : 1 = false;
		bool modLike : 1 = false;
		bool isSafe : 1 = false;
		bool exported : 1 = false;
		//bool exportedToTests : 1 = false;
	};
	struct GenModPath
	{
		ModPath mp;
		std::string name;

		std::unordered_map<std::string, LocalObjId> name2Obj;
		std::unordered_map<LocalObjId, GenObj> objs;//if not in list, then resolved

		bool resolved : 1 = false;//no need to check objs, dont add new things

		std::shared_mutex lock; // unique, when adding / writing objs
	};
	struct GenState
	{
		MidState out;//Only write to this, if there are 0 unresolved things

		std::unordered_map<ModPath, ModPathId> mp2Id;
		std::unordered_map<ModPathId, GenModPath> unresolvedMps;

		// unique, only when adding stuff to root fields
		// OR, when adding to any part of out.
		// TODO: maybe relax that, by tagging the unresolved thing, and forcing data requests to get a unique lock?
		std::shared_mutex lock;
	};
}