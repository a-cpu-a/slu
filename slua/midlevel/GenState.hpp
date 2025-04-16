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
	struct GenModPath
	{
		std::unordered_map<std::string, LocalObjId> name2Obj;

		std::shared_mutex lock; // unique, when adding objs
	};
	struct GenState
	{
		MidState out;//Only write to this, if there are 0 unresolved things

		std::unordered_map<ModPath, ModPathId> mp2Id;
		std::unordered_map<ModPathId, GenModPath> modPaths;

		// unique, only when adding stuff to root fields
		// OR, when adding to root fields of out.
		std::shared_mutex lock;


		ModPathId getMpId(const ModPath& mp)
		{
			if (!mp2Id.contains(mp))
			{
				ModPathId id = { out.modPaths.size() };
				mp2Id[mp] = id;
				modPaths[id];// Insert
				return id;
			}
			return mp2Id[mp];
		}

		ModPathId getMpIdLock(const ModPath& mp)
		{
			std::unique_lock _(lock);
			return getMpId(mp);
		}
	};
}