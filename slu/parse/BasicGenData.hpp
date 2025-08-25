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

import slu.ast.state;
import slu.parse.input;
#include <slu/lang/Mpc.hpp>

namespace slu::parse
{

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

	struct _ew_string_haah {
		using is_transparent = void;
		auto operator()(const auto& a) const noexcept {
			return std::hash<std::string_view>{}((std::string_view)a);
		}
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

	struct TraitFn
	{
		lang::MpItmId name;
		std::vector<ResolvedType> args;
		ResolvedType ret;
		lang::ExportData exported;

	};
	namespace ItmType
	{
		using namespace std::string_view_literals;

		//Applied to anon/synthetic/private? stuff
		struct Unknown {
			constexpr static std::string_view NAME = "unknown"sv;
		};

		struct Fn
		{
			constexpr static std::string_view NAME = "function"sv;
			std::string abi;
			ResolvedType ret;
			std::vector<ResolvedType> args;
			std::vector<parse::LocalId> argLocals;//Only for non decl functions
			lang::ExportData exported;
		};
		struct NamedTypeImpls
		{
			constexpr static std::string_view NAME = "unknown-type"sv;
			std::vector<lang::MpItmId> impls;
		};
		struct TypeFn : Fn, NamedTypeImpls
		{
			constexpr static std::string_view NAME = "type-function"sv;
		};
		struct ArglessTypeVar : NamedTypeImpls
		{
			constexpr static std::string_view NAME = "type"sv;
			ResolvedType ty;
		};
		//like basic global vars in c++!
		struct GlobVar
		{
			constexpr static std::string_view NAME = "static-var"sv;
			ResolvedType ty;
		};
		//No precomputation for now, instead make them act as macros
		struct ConstVar
		{
			constexpr static std::string_view NAME = "const-var"sv;
			ResolvedType ty;
			parse::Expr value;
		};
		struct Trait
		{
			constexpr static std::string_view NAME = "trait"sv;
			std::vector<lang::MpItmId> consts;
			std::vector<TraitFn> fns;
			lang::ExportData exported;
			//TODO: where clauses
		};
		struct Impl
		{
			constexpr static std::string_view NAME = "impl"sv;
			lang::ExportData exported;
			//TODO: impl data (params, impl-consts, impl-fn's, where clauses)
		};
		struct Alias//TODO something for ::* use's
		{
			constexpr static std::string_view NAME = "alias"sv;
			lang::MpItmId usedThing;
		};
		struct Module{
			constexpr static std::string_view NAME = "module"sv;
		};
	}
	using Itm = std::variant<
		ItmType::Unknown,
		ItmType::Fn,
		ItmType::NamedTypeImpls, // Used temporarily while a type is not defined, but does have impl's
		ItmType::TypeFn,
		ItmType::ArglessTypeVar,
		ItmType::GlobVar,
		ItmType::ConstVar,
		ItmType::Trait,
		ItmType::Impl,
		ItmType::Alias,
		ItmType::Module
	>;

	using MpItmName2Obj = std::unordered_map<std::string, lang::LocalObjId, _ew_string_haah, _ew_string_eq>;
	struct BasicModPathData
	{
		lang::ModPath path;
		MpItmName2Obj name2Id;
		std::vector<std::string> id2Name;
		std::vector<parse::Itm> id2Itm;

		void addItm(lang::LocalObjId obj,Itm&& v)
		{
			if (id2Itm.size() <= obj.val)
				id2Itm.resize(obj.val + 1);
			id2Itm[obj.val] = std::move(v);
		}

		lang::LocalObjId at(const std::string_view name) const {
			return name2Id.find(name)->second;
		}
		lang::LocalObjId get(std::string_view name)
		{
			auto p = name2Id.find(name);
			if (p == name2Id.end())
			{
				const size_t res = id2Name.size();

				name2Id[std::string(name)] = { res };
				id2Name.emplace_back(name);

				return { res };
			}
			return p->second;
		}
		lang::LocalObjId get(std::string&& name)
		{
			auto p = name2Id.find(name);
			if (p == name2Id.end())
			{
				const size_t res = id2Name.size();

				name2Id[std::string(name)] = { res };
				id2Name.emplace_back(std::move(name));

				return { res };
			}
			return p->second;
		}

		BasicModPathData() = default;
		BasicModPathData(lang::ModPath&& path) :path(std::move(path)) {}
		BasicModPathData(const BasicModPathData&) = delete;
		BasicModPathData(BasicModPathData&&) = default;
		BasicModPathData& operator=(BasicModPathData&&) = default;
	};
	using Mp2MpIdMap = std::unordered_map<lang::ModPath, lang::ModPathId, lang::HashModPathView, lang::EqualModPathView>;
	template<size_t N>
	inline void initMpData(Mp2MpIdMap& mp2Id,BasicModPathData& mp, const mpc::MpcMp<N>& data,size_t itmCount)
	{
		mp.path.resize(data.mp.size());


		for (size_t i = 0; i < data.mp.size(); i++)
			mp.path[i] = std::string(data.mp[i]);
		//mp.id2Itm.reserve(itmCount);
		mp.id2Name.reserve(itmCount);
		mp.name2Id.reserve(itmCount);

		if (!data.mp.empty())
			mp2Id[mp.path] = data.id;
	}
	struct BasicMpDbData
	{
		Mp2MpIdMap mp2Id;
		std::vector<parse::BasicModPathData> mps;

		BasicMpDbData() {
			using namespace std::string_view_literals;
			mps.resize(mpc::MP_COUNT);
#define _Slu_INIT_MP(ARG_MP) \
	size_t i_##ARG_MP = 0; \
	BasicModPathData& mp_##ARG_MP = mps[::slu::mpc::MP_##ARG_MP.idx()]; \
	initMpData(mp2Id, mp_##ARG_MP, \
		::slu::mpc::MP_##ARG_MP, \
		::slu::mpc::MP_ITM_COUNT_##ARG_MP \
	); \
	Slu_##ARG_MP##_ITEMS(;)
#define _Slu_HANDLE_ITEM(ARG_MP,_VAR,_STR) \
	mp_##ARG_MP.id2Name.emplace_back(_STR##sv); \
	mp_##ARG_MP.name2Id.emplace(_STR##sv,i_##ARG_MP++)
#define _X(_VAR,_STR) _Slu_HANDLE_ITEM(POOLED,_VAR,_STR)
			_Slu_INIT_MP(POOLED);
#undef _X
#define _X(_VAR,_STR) _Slu_HANDLE_ITEM(STD,_VAR,_STR)
			_Slu_INIT_MP(STD);
#undef _X
#define _X(_VAR,_STR) _Slu_HANDLE_ITEM(STD_BOOL,_VAR,_STR)
			_Slu_INIT_MP(STD_BOOL);
#undef _X
#undef _Slu_HANDLE_ITEM
#undef _Slu_INIT_MP
		}

		const Itm& getItm(const lang::MpItmId name) const {
			return mps[name.mp.id].id2Itm[name.id.val];
		}

		//Returns empty if not found
		const lang::PoolString getPoolStr(std::string_view txt) const {
			auto p = mps[mpc::MP_UNKNOWN.idx()].name2Id.find(txt);
			if (p == mps[mpc::MP_UNKNOWN.idx()].name2Id.end())
				return lang::PoolString::newEmpty();
			return lang::PoolString{ p->second };
		}

		lang::ModPath getMp(const lang::MpItmId name)const
		{
			const BasicModPathData& data = mps[name.mp.id];
			lang::ModPath res;
			res.reserve(data.path.size() + 1);
			res.insert(res.end(), data.path.begin(), data.path.end());
			res.push_back(data.id2Name[name.id.val]);
			return res;
		}

		lang::MpItmId getItm(const lang::AnyMp auto& path) const
		{
			if (path.size() == 1)
			{
				throw std::runtime_error("TODO: crate values: get item from a path with 1 element");
			}
			lang::MpItmId res;
			res.mp = mp2Id.find(path.subspan(0, path.size() - 1))->second;
			res.id = mps[res.mp.id].at(path.back());
			return res;
		}
		lang::MpItmId getItm(const std::initializer_list<std::string_view>& path) const {
			return getItm((lang::ViewModPathView)path);
		}

		std::string_view getSv(const lang::PoolString v) const {
			if (v.val == SIZE_MAX)
				return {};//empty
			return mps[mpc::MP_UNKNOWN.idx()].id2Name[v.val];
		}
	};
	template<class T, bool followAlias=false>
	inline const T& getItm(const BasicMpDbData& mpDb,const lang::MpItmId name)
	{
		return *ezmatch(mpDb.getItm(name))(
		[&]<class T2>(const T2& var)->const T* {
			throw std::runtime_error("Expected item type " + std::string(T::NAME) + ", but got " + std::string(T2::NAME));
		},
		varcase(const ItmType::Alias&)->const T* requires(followAlias) {
			return &getItm<T, followAlias>(mpDb, var.usedThing);
		},
		varcase(const T&)->const T* {
			return &var;
		}
		);
	}
	inline lang::MpItmId resolveAlias(const BasicMpDbData& mpDb,const lang::MpItmId name)
	{
		return ezmatch(mpDb.getItm(name))(
		varcase(const auto&) {
			return name;
		},
		varcase(const ItmType::Alias&) {
			return resolveAlias(mpDb, var.usedThing);
		}
		);
	}
	struct BasicMpDb
	{
		BasicMpDbData* data;

		bool isUnknown(lang::MpItmId n) const
		{
			if (n.mp.id == mpc::MP_UNKNOWN.idx())
				return true;//Hardcoded as always unknown.
			//Else: check if first part is empty
			return data->mps[n.mp.id].path[0].empty();
		}
		lang::PoolString poolStr(const std::string_view name) {
			return data->mps[mpc::MP_UNKNOWN.idx()].get(name);
		}
		lang::MpItmId resolveUnknown(const std::string_view name) {
			return lang::MpItmId{poolStr(name), { mpc::MP_UNKNOWN.idx() }};
		}

		template<bool unknown>
		lang::ModPathId get(const lang::ModPathView path)
		{
			if (!data->mp2Id.contains(path))
			{
				const size_t res = data->mps.size();

				data->mp2Id.emplace(lang::ModPath(path.begin(), path.end()), res);

				if constexpr (unknown)
				{
					lang::ModPath tmp;
					tmp.reserve(1 + path.size());
					tmp.push_back("");
					tmp.insert(tmp.end(),path.begin(), path.end());
					data->mps.emplace_back(std::move(tmp));
				}
				else
					data->mps.emplace_back(lang::ModPath(path.begin(), path.end()));

				return { res };
			}
			return data->mp2Id.find(path)->second;
		}

		lang::MpItmId getItm(const lang::ModPathView path)
		{
			if (path.size() == 1)
			{
				throw std::runtime_error("TODO: crate values: get item from a path with 1 element");
			}
			lang::MpItmId res;
			res.mp = get<false>(path.subspan(0,path.size()-1));
			res.id = data->mps[res.mp.id].get(path.back());
			return res;
		}

		std::string_view asSv(const lang::MpItmId v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return data->mps[v.mp.id].id2Name[v.id.val];
		}
		std::string_view asSv(const lang::PoolString v) const {
			if (v.val == SIZE_MAX)
				return {};//empty
			return data->mps[mpc::MP_UNKNOWN.idx()].id2Name[v.val];
		}
		lang::ViewModPath asVmp(const lang::MpItmId v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			const BasicModPathData& mp = data->mps[v.mp.id];

			lang::ViewModPath res;
			res.reserve(mp.path.size() + 1);

			for (const std::string& s : mp.path)
			{
				if(!s.starts_with('$'))
					res.push_back(s);
			}
			res.push_back(mp.id2Name[v.id.val]);

			return res;
		}
	};

	std::string_view _fwdConstructBasicMpDbAsSv(BasicMpDbData* data, lang::MpItmId thiz){
		return BasicMpDb{ data }.asSv(thiz);
	}

	lang::ViewModPath _fwdConstructBasicMpDbAsVmp(BasicMpDbData* data, lang::MpItmId thiz){
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
		std::vector<LocalsV<isSlu>> localsStack;
		BasicMpDb mpDb;
		std::vector<BasicGenScopeV<isSlu>> scopes;
		std::vector<parse::LocalId> anonScopeCounts;
		lang::ModPath totalMp;

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

		std::string_view asSv(const lang::PoolString id) const {
			return mpDb.asSv(id);
		}
		std::string_view asSv(const lang::MpItmId id) const {
			return mpDb.asSv(id);
		}
		lang::ViewModPath asVmp(const lang::MpItmId v) const {
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
		constexpr void pushAnonScope(const ast::Position start)
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
		constexpr void pushUnScope(const ast::Position start,const bool isGlobal)
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
		constexpr void pushScope(const ast::Position start,std::string&& name) {
			//addLocalObj(name);

			totalMp.push_back(std::move(name));
			scopes.push_back({ NORMAL_SCOPE });
			scopes.back().res.start = start;
			anonScopeCounts.emplace_back(0);
		}
		constexpr LocalsV<isSlu> popLocalScope() {
			auto res = std::move(localsStack.back());
			localsStack.pop_back();
			return std::move(res);
		}
		BlockV<isSlu> popScope(const ast::Position end) {
			BlockV<isSlu> res = std::move(scopes.back().res);
			res.end = end;
			if constexpr(isSlu)
				res.mp = mpDb.template get<false>(totalMp);
			scopes.pop_back();
			totalMp.pop_back();
			anonScopeCounts.pop_back();
			return res;
		}
		BlockV<isSlu> popUnScope(const ast::Position end) {
			BlockV<isSlu> res = std::move(scopes.back().res);
			res.end = end;
			bool isGlobal = scopes.back().anonId == GLOBAL_SCOPE;
			scopes.pop_back();
			const LocalId nextAnonId = anonScopeCounts.back();
			anonScopeCounts.pop_back();
			if (isGlobal)
			{
				if constexpr (isSlu)
					res.mp = mpDb.template get<false>(totalMp);
			}
			else
			{
				anonScopeCounts.back() = nextAnonId;//Shared counter
			}
			return res;
		}
		void scopeReturn(RetType ty) {
			scopes.back().res.retTy = ty;
		}
		// Make sure to run no args `scopeReturn()` first!
		void scopeReturn(ExprList&& expList) {
			scopes.back().res.retExprs = std::move(expList);
		}

		constexpr void addStat(const ast::Position place,StatDataV<isSlu>&& data){
			Stat stat = { std::move(data) };
			stat.place = place;
			scopes.back().res.statList.emplace_back(std::move(stat));
		}
		constexpr lang::MpItmId addLocalObj(const std::string& name)
		{
			size_t mpPopCount = 0;
			for (auto& i : std::views::reverse(scopes))
			{
				if (i.anonId != UNSCOPE)
				{
					i.objs.push_back(name);
					if constexpr (isSlu)
					{
						auto mpView = lang::ModPathView(totalMp).subspan(0, totalMp.size() - mpPopCount);
						lang::ModPathId mp = mpDb.template get<false>(mpView);
						lang::LocalObjId id = mpDb.data->mps[mp.id].get(name);
						return lang::MpItmId{id, mp};
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
				if (scope.anonId == UNSCOPE)
					continue;

				for (const std::string& var : scope.objs)
				{
					if (var == name)
						return scopeRevId;
				}
				if (scope.anonId == GLOBAL_SCOPE)
					break;//Dont look into other modules

				scopeRevId++;
			}
			return {};
		}

		constexpr DynLocalOrNameV<isSlu> resolveNameOrLocal(const std::string& name)
		{// Check if its local
			if constexpr (isSlu)
			{
				//either known local being indexed ORR unknown(potentially from a `use ::*`)
				if (!localsStack.empty())
				{
					size_t id = 0;
					for (auto& i : localsStack.back().names)
					{
						if(mpDb.data->mps[i.mp.id].id2Name[i.id.val]==name)
							return LocalId{ id };
						id++;
					}
				}
				const std::optional<size_t> v = resolveLocalOpt(name);
				if (v.has_value())
				{
					lang::ModPathId mp = mpDb.template get<false>(
						lang::ModPathView(totalMp).subspan(0, totalMp.size() -  *v)
					);
					lang::LocalObjId id = mpDb.data->mps[mp.id].get(name);
					return lang::MpItmId{id, mp};
				}
			}
			return resolveUnknown(name);
		}
		constexpr lang::MpItmId resolveName(const std::string& name)
		{
			if constexpr (isSlu)
			{
				return ezmatch(resolveNameOrLocal(name))(
					varcase(const parse::LocalId) { return localsStack.back().names[var.v]; },
					varcase(const lang::MpItmId) {return var;	}
					);
			}
			else
				return resolveNameOrLocal(name);
		}
		constexpr lang::MpItmId resolveRootName(const lang::ModPath& name) {
			return mpDb.getItm(name);// Create if needed, and return it
		}
		constexpr size_t countScopes() const
		{
			size_t val=0;
			for (const auto& i : std::views::reverse(scopes))
			{
				if (i.anonId == GLOBAL_SCOPE)
					break;
				if (i.anonId != UNSCOPE)
					val++;
			}
			return val;
		}
		constexpr lang::MpItmId resolveName(const lang::ModPath& name)
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
				lang::ModPath mpSum;
				mpSum.reserve((totalMp.size() - *v) + (name.size() - 2));

				for (size_t i = 0; i < totalMp.size() - *v; i++)
					mpSum.push_back(totalMp[i]);
				for (size_t i = remFirst?1:0; i < name.size()-1; i++)
					mpSum.push_back(name[i]);

				lang::ModPathId mp = mpDb.template get<false>(lang::ModPathView(mpSum));

				lang::LocalObjId id = mpDb.data->mps[mp.id].get(name.back());
				return lang::MpItmId{id,mp};
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
				stack.names.push_back(n);
				return parse::LocalId(stack.names.size() - 1);
			}
			else
				return n;
		}
		template<bool isLocal>
		constexpr LocalOrNameV<isSlu, isLocal> resolveNewSynName()
		{
			const size_t id = anonScopeCounts.back().v++;
			const std::string name = getAnonName(id);
			auto n = addLocalObj(name);
			if constexpr (isLocal)
			{
				auto& stack = localsStack.back();
				stack.names.push_back(n);
				return parse::LocalId(stack.names.size() - 1);
			}
			else
				return n;
		}
		// .XXX, XXX, :XXX
		constexpr lang::MpItmId resolveUnknown(const std::string& name)
		{
			if constexpr(isSlu)
				return mpDb.resolveUnknown(name);
			else
			{
				lang::LocalObjId id = mpDb.get(name);
				return lang::MpItmId{id};
			}
		}
		lang::PoolString poolStr(std::string&& name)
		{
			if constexpr (isSlu)
				return mpDb.data->mps[mpc::MP_UNKNOWN.idx()].get(std::move(name));
			else
				return mpDb.get(std::move(name));
		}
		constexpr lang::MpItmId resolveUnknown(const lang::ModPath& name)
		{
			lang::ModPathId mp = mpDb.template get<true>(
				lang::ModPathView(name).subspan(0, name.size() - 1) // All but last elem
			);
			lang::LocalObjId id = mpDb.data->mps[mp.id].get(name.back());
			return lang::MpItmId{id,mp};
		}
		constexpr lang::MpItmId resolveEmpty()
		{
			return lang::MpItmId::newEmpty();
		}
	};
}