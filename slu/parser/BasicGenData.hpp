/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <ranges>
#include <unordered_map>
#include <ranges>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>

namespace slu::parse
{
	using lang::LocalObjId;
	using lang::ModPathId;
	using lang::AnyMp;


	const size_t NORMAL_SCOPE = SIZE_MAX;
	const size_t UNSCOPE = SIZE_MAX-1;
	const size_t GLOBAL_SCOPE = SIZE_MAX-2;

	/*
		Names starting with $ are anonymous, and are followed by 8 raw bytes (representing anon name id)
		(All of those are not to be used by other modules, and are local only)

		When converting anonymous names into text, use this form: $(hex form of bytes, no leading zeros except for just 0)

		If a host asks for one of these names as text, then use one of these forms:
			Same as for normal text or
			__SLUANON__(hex form of bytes, same as for normal text)

		(Hex forms in lowercase)
	*/
	constexpr std::string getAnonName(const size_t anonId)
	{
		_ASSERT(anonId != NORMAL_SCOPE && anonId != UNSCOPE && anonId != GLOBAL_SCOPE);
		std::string name(1 + sizeof(size_t), '$');
		name[1] = uint8_t((anonId & 0xFF00000000000000) >> 56);
		name[2] = uint8_t((anonId & 0xFF000000000000) >> 48);
		name[3] = uint8_t((anonId & 0xFF0000000000) >> 40);
		name[4] = uint8_t((anonId & 0xFF00000000) >> 32);
		name[5] = uint8_t((anonId & 0xFF000000) >> 24);
		name[6] = uint8_t((anonId & 0xFF0000) >> 16);
		name[7] = uint8_t((anonId & 0xFF00) >> 8);
		name[8] = uint8_t((anonId & 0xFF) >> 0);
		return name;
	}

	struct _ew_string_haah:std::hash<std::string>, std::hash<std::string_view> {
		using is_transparent = void;
	};
	struct _ew_string_eq {
		using is_transparent = void;
		constexpr bool operator()(const std::string& a, const std::string& b) const {
			return a == b;
		}
		constexpr bool operator()(const std::string_view a, const std::string_view b) const {
			return a == b;
		}
		constexpr bool operator()(const std::string& a, const std::string_view b) const {
			return a == b;
		}
		constexpr bool operator()(const std::string_view a, const std::string& b) const {
			return a == b;
		}
	};

	struct BasicModPathData
	{
		ModPath path;
		std::unordered_map<std::string, LocalObjId,_ew_string_haah,_ew_string_eq> name2Id;
		std::unordered_map<size_t, std::string> id2Name;

		LocalObjId at(const std::string_view name) const {
			return name2Id.find(name)->second;
		}
		LocalObjId get(const std::string_view name)
		{
			auto p = name2Id.find(name);
			if (p == name2Id.end())
			{
				const size_t res = name2Id.size();

				name2Id[std::string(name)] = { res };
				id2Name[res] = std::string(name);

				return { res };
			}
			return p->second;
		}
	};
	struct LuaMpDb
	{
		std::unordered_map<std::string, LocalObjId> name2Id;
		std::unordered_map<size_t, std::string> id2Name;

		LocalObjId get(const std::string& v)
		{
			if (!name2Id.contains(v))
			{
				const size_t res = name2Id.size();

				name2Id[v] = { res };
				id2Name[res] = v;

				return { res };
			}
			return name2Id[v];
		}

		std::string_view asSv(const MpItmIdV<false> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return id2Name.at(v.id.val);
		}
		lang::ViewModPath asVmp(const MpItmIdV<false> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return { id2Name.at(v.id.val) };
		}
	};
	struct BasicMpDbData
	{
		std::unordered_map<ModPath, ModPathId, lang::HashModPathView, lang::EqualModPathView> mp2Id;
		std::vector<BasicModPathData> mps = { {} };//Add 0, the unknown one

		ModPath getMp(const MpItmIdV<true> name)const 
		{
			const BasicModPathData& data = mps[name.mp.id];
			ModPath res;
			res.reserve(data.path.size() + 1);
			res.insert(res.end(), data.path.begin(), data.path.end());
			res.push_back(data.id2Name.at(name.id.val));
			return res;
		}

		MpItmIdV<true> getItm(const AnyMp auto& path) const
		{
			if (path.size() == 1)
			{
				throw std::runtime_error("TODO: crate values: get item from a path with 1 element");
			}
			MpItmIdV<true> res;
			res.mp = mp2Id.find(path.subspan(0, path.size() - 1))->second;
			res.id = mps[res.mp.id].at(path.back());
			return res;
		}
		MpItmIdV<true> getItm(const std::initializer_list<std::string_view>& path) const {
			return getItm((lang::ViewModPathView)path);
		}
	};
	struct BasicMpDb
	{
		BasicMpDbData* data;

		

		template<bool unknown>
		ModPathId get(const ModPathView path)
		{
			if (!data->mp2Id.contains(path))
			{
				const size_t res = data->mps.size();

				data->mp2Id.emplace(ModPath(path.begin(), path.end()), res);

				if constexpr (unknown)
				{
					ModPath tmp;
					tmp.reserve(1 + path.size());
					tmp.push_back("");
					tmp.insert(tmp.end(),path.begin(), path.end());
					data->mps.emplace_back(std::move(tmp));
				}
				else
					data->mps.emplace_back(ModPath(path.begin(), path.end()));

				return { res };
			}
			return data->mp2Id.find(path)->second;
		}
		MpItmIdV<true> getItm(const ModPathView path)
		{
			if (path.size() == 1)
			{
				throw std::runtime_error("TODO: crate values: get item from a path with 1 element");
			}
			MpItmIdV<true> res;
			res.mp = get<false>(path.subspan(0,path.size()-1));
			res.id = data->mps[res.mp.id].get(path.back());
			return res;
		}

		std::string_view asSv(const MpItmIdV<true> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return data->mps[v.mp.id].id2Name.at(v.id.val);
		}
		lang::ViewModPath asVmp(const MpItmIdV<true> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			const BasicModPathData& mp = data->mps[v.mp.id];

			lang::ViewModPath res;
			res.reserve(mp.path.size() + 1);

			for (const std::string& s : mp.path)
			{
				if(s.front()!='$')
					res.push_back(s);
			}
			res.push_back(mp.id2Name.at(v.id.val));

			return res;
		}
	};

	std::string_view _fwdConstructBasicMpDbAsSv(BasicMpDbData* data, MpItmIdV<true> thiz){
		return BasicMpDb{ data }.asSv(thiz);
	}

	lang::ViewModPath _fwdConstructBasicMpDbAsVmp(BasicMpDbData* data, MpItmIdV<true> thiz){
		return BasicMpDb{ data }.asVmp(thiz);
	}

	struct GenSafety
	{
		bool isSafe : 1 = false;
		bool forPop : 1 = false;
	};
	template<bool isSlu>
	struct BasicGenScopeV
	{
		size_t anonId;//size_max -> not anon

		std::vector<std::string> objs;

		std::vector<GenSafety> safetyList;

		BlockV<isSlu> res;

	};
	template<bool isSlu>
	struct BasicGenDataV
	{
		//ParsedFileV<isSlu> out; //TODO_FOR_COMPLEX_GEN_DATA: field is ComplexOutData&, and needs to be obtained from shared mutex
		std::vector<LocalsV<isSlu>> localsStack;
		Sel<isSlu, LuaMpDb, BasicMpDb> mpDb;
		std::vector<BasicGenScopeV<isSlu>> scopes;
		std::vector<LocalId> anonScopeCounts;
		ModPath totalMp;

		/*
		All local names (no ::'s) are defined in THIS file, or from a `use ::*` (potentialy std::prelude::*)
		This means any 'use' things that reference ??? stuff, can only be other global crates OR a sub-path of a `use ::*`
		
		Since we dont know if 'mylib' is a crate, or a module inside something that was `use ::*`-ed,
		we have to wait until all ::* modules have been parsed.
		OR, we have to let another pass figure it out. (The simpler option)

		Recursive ::* uses, might force the multi-pass aproach?
		Or would be impossible.
		Which is why ::* should enforce order and no recursive use.

		How to handle 2 `use ::*`'s?

		If there are 2 star uses, is it required to union both of them, into 1 symbol id?
		Are we forced to make a new symbol after every combination of star uses? No (VVV)
		Is it better to just use unknown_modpath?
		^ Yes, doing so, will simplify the ability to add or remove the default used files

		*/

		std::string_view asSv(const MpItmIdV<isSlu> id) const {
			return mpDb.asSv(id);
		}
		lang::ViewModPath asVmp(const MpItmIdV<isSlu> v) const {
			return { mpDb.asVmp(v)};
		}

		constexpr void pushUnsafe() {
			scopes.back().safetyList.emplace_back(false, true);
		}
		constexpr void popSafety()
		{
			std::vector<GenSafety>& safetyList = scopes.back().safetyList;
			size_t popCount = 1;

			for (const GenSafety gs : std::views::reverse(safetyList))
			{
				if (gs.forPop)
					break;
				popCount++;
			}
			safetyList.erase(safetyList.end()-popCount, safetyList.end());
		}
		constexpr void setUnsafe()
		{
			if(!scopes.back().safetyList.empty())
			{
				GenSafety& gs = scopes.back().safetyList.back();
				if (!(gs.forPop || gs.isSafe))
					return;
			}
			scopes.back().safetyList.emplace_back(false);
		}
		constexpr void setSafe()
		{
			if (!scopes.back().safetyList.empty())
			{
				GenSafety& gs = scopes.back().safetyList.back();
				if (!(gs.forPop || !gs.isSafe))
					return;
			}
			scopes.back().safetyList.emplace_back(true);
		}

		constexpr void pushLocalScope() {
			localsStack.emplace_back();
		}

		//For impl, lambda, scope, doExpr, things named '_'
		constexpr void pushAnonScope(const Position start)
		{
			const size_t id = anonScopeCounts.back().v++;
			const std::string name = getAnonName(id);
			addLocalObj(name);

			totalMp.push_back(name);
			scopes.push_back({id});
			scopes.back().res.start = start;
			anonScopeCounts.emplace_back(0);
		}
		//For extern/unsafe blocks
		constexpr void pushUnScope(const Position start,const bool isGlobal)
		{
			const size_t id = isGlobal ? GLOBAL_SCOPE : UNSCOPE;

			scopes.push_back({ id });
			scopes.back().res.start = start;
			if (isGlobal)
				anonScopeCounts.emplace_back(0);
			else
				anonScopeCounts.push_back(anonScopeCounts.back());
		}
		//For func, macro, inline_mod, type?, ???
		constexpr void pushScope(const Position start,const std::string& name) {
			//addLocalObj(name);

			totalMp.push_back(name);
			scopes.push_back({ NORMAL_SCOPE });
			scopes.back().res.start = start;
			anonScopeCounts.emplace_back(0);
		}
		constexpr LocalsV<isSlu> popLocalScope() {
			auto res = std::move(localsStack.back());
			localsStack.pop_back();
			return std::move(res);
		}
		BlockV<isSlu> popScope(const Position end) {
			BlockV<isSlu> res = std::move(scopes.back().res);
			res.end = end;
			res.mp = mpDb.get<false>(totalMp);
			scopes.pop_back();
			totalMp.pop_back();
			anonScopeCounts.pop_back();
			return res;
		}
		BlockV<isSlu> popUnScope(const Position end) {
			BlockV<isSlu> res = std::move(scopes.back().res);
			res.end = end;
			bool isGlobal = scopes.back().anonId == GLOBAL_SCOPE;
			scopes.pop_back();
			const LocalId nextAnonId = anonScopeCounts.back();
			anonScopeCounts.pop_back();
			if (isGlobal)
				res.mp = mpDb.get<false>(totalMp);
			else
			{
				anonScopeCounts.back() = nextAnonId;//Shared counter
			}
			return res;
		}
		void scopeReturn() {
			scopes.back().res.hadReturn = true;
		}
		// Make sure to run no args `scopeReturn()` first!
		void scopeReturn(ExpListV<isSlu>&& expList) {
			scopes.back().res.retExprs = std::move(expList);
		}

		constexpr void addStat(const Position place,StatementDataV<isSlu>&& data){
			StatementV<isSlu> stat = { std::move(data) };
			stat.place = place;
			scopes.back().res.statList.emplace_back(std::move(stat));
		}
		constexpr MpItmIdV<isSlu> addLocalObj(const std::string& name)
		{
			size_t mpPopCount = 0;
			for (auto& i : std::views::reverse(scopes))
			{
				if (i.anonId != UNSCOPE)
				{
					i.objs.push_back(name);
					if constexpr (isSlu)
					{
						auto mpView = ModPathView(totalMp).subspan(0, totalMp.size() - mpPopCount);
						ModPathId mp = mpDb.template get<false>(mpView);
						LocalObjId id = mpDb.data->mps[mp.id].get(name);
						return MpItmIdV<true>{id, mp};
					}
					else
						return resolveUnknown(name);
					mpPopCount++;
				}
			}
			throw std::runtime_error("No scope to add local object to");
		}

		constexpr std::optional<size_t> resolveLocalOpt(const std::string& name)
		{
			size_t scopeRevId = 0;
			for (const BasicGenScopeV<isSlu>& scope : std::views::reverse(scopes))
			{
				scopeRevId++;

				for (const std::string& var : scope.objs)
				{
					if (var == name)
						return scopeRevId;
				}
			}
			return {};
		}
		constexpr MpItmIdV<isSlu> resolveName(const std::string& name)
		{// Check if its local
			if constexpr (isSlu)
			{
				//either known local being indexed ORR unknown(potentially from a `use ::*`)
				if (!localsStack.empty())
				{
					for (auto& i : localsStack.back())
					{
						if(mpDb.data->mps[i.mp.id].id2Name[i.id.val]==name)
							return i;
					}
				}
				const std::optional<size_t> v = resolveLocalOpt(name);
				if (v.has_value())
				{
					ModPathId mp = mpDb.template get<false>(
						ModPathView(totalMp).subspan(0, totalMp.size() - *v)
					);
					LocalObjId id = mpDb.data->mps[mp.id].get(name);
					return MpItmIdV<true>{id, mp};
				}
			}
			return resolveUnknown(name);
		}
		constexpr MpItmIdV<isSlu> resolveRootName(const ModPath& name) {
			return mpDb.getItm(name);// Create if needed, and return it
		}
		constexpr size_t countScopes() const
		{
			size_t val=0;
			for (const auto& i : scopes)
			{
				if (i.anonId != UNSCOPE && i.anonId!= GLOBAL_SCOPE)
					val++;
			}
			return val;
		}
		constexpr MpItmIdV<isSlu> resolveName(const ModPath& name)
		{
			if (name.size() == 1)
				return resolveName(name[0]);//handles self implicitly!!!

			//either known local being indexed, super, crate, self ORR unknown(potentially from a `use ::*`)

			std::optional<size_t> v;
			bool remFirst = true;
			if (name[0] == "self")
				v = countScopes();
			else if (name[0] == "super")
				v = countScopes()+1;//Pop all new ones + self
			else if (name[0] == "crate")
				v = totalMp.size() - 1;//All but last
			else
			{
				remFirst = false;
				v = resolveLocalOpt(name[0]);
			}

			if (v.has_value())
			{
				ModPath mpSum;
				mpSum.reserve((totalMp.size() - *v) + (name.size() - 2));

				for (size_t i = 0; i < totalMp.size() - *v; i++)
					mpSum.push_back(totalMp[i]);
				for (size_t i = remFirst?1:0; i < name.size()-1; i++)
					mpSum.push_back(name[i]);

				ModPathId mp = mpDb.template get<false>(ModPathView(mpSum));

				LocalObjId id = mpDb.data->mps[mp.id].get(name.back());
				return MpItmIdV<true>{id,mp};
			}
			return resolveUnknown(name);
		}
		constexpr bool inGlobalLand() const {
			return localsStack.empty();
		}
		template<bool isLocal>
		constexpr LocalOrNameV<isSlu,isLocal> resolveNewName(const std::string& name)
		{
			auto n = addLocalObj(name);
			if constexpr (isLocal)
			{
				auto& stack = localsStack.back();
				stack.push_back(n);
				return LocalId(stack.size() - 1);
			}
			else
				return n;
		}
		// .XXX, XXX, :XXX
		constexpr MpItmIdV<isSlu> resolveUnknown(const std::string& name)
		{
			if constexpr(isSlu)
			{
				LocalObjId id = mpDb.data->mps[0].get(name);
				return MpItmIdV<true>{id, { 0 }};
			}
			else
			{
				LocalObjId id = mpDb.get(name);
				return MpItmIdV<false>{id};
			}
		}
		constexpr MpItmIdV<isSlu> resolveUnknown(const ModPath& name)
		{
			ModPathId mp = mpDb.template get<true>(
				ModPathView(name).subspan(0, name.size() - 1) // All but last elem
			);
			LocalObjId id = mpDb.data->mps[mp.id].get(name.back());
			return MpItmIdV<true>{id,mp};
		}
		constexpr MpItmIdV<isSlu> resolveEmpty()
		{
			return MpItmIdV<isSlu>::newEmpty();
		}
	};
}