/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>

#include <slu/lang/BasicState.hpp>

namespace slu::comp
{
	//
	// All views / spans are only required to live as long as the call.
	// The compiler wont emit `../`, so there is no need to sanitize them for that (except by you in rootPaths)
	// Your paths must not contain `\`
	// If one of your callbacks is thread-unsafe, add some mutex, because all of them will be called by many threads.
	//

	

	//TODO: implement compiler plugins, maybe redesign the api too.

	//Used to implement external namespaces (lua::ext::*)
	struct CompPlugin
	{
		// If it would need to call to some dll, or external code, it is likely not safe
		bool comptimeSafe : 1 = false;

		lang::ModPathView root;

		//TODO: choose a type for the list (string?) (mod path?)
		using GetItemListFn = std::vector<int>(*)(const lang::ModPathView& path);
		GetItemListFn getItemListPtr;

		using ItemBoolFn = bool(*)(const lang::ModPathView& path);
		ItemBoolFn itemExistsPtr;
	};
	struct CompCfg
	{
		//you need to add a newline yourself, if you need to (use println or something)
		//Note: msg may contain nulls
		using LogFn = void(*)(const std::string_view msg);
		LogFn logPtr;

		using GetFileContentsFn = std::optional<std::vector<uint8_t>>(*)(const std::string_view path);
		GetFileContentsFn getFileContentsPtr;

		using FileBoolFn = bool(*)(const std::string_view path);
		FileBoolFn fileExistsPtr;
		FileBoolFn isFolderPtr;
		FileBoolFn isSymLinkPtr;

		//file paths inside a path, return empty list if path is invalid or doesnt exist
		using GetFileListFn = std::vector<std::string>(*)(const std::string_view path);
		GetFileListFn getFileListPtr;
		GetFileListFn getFileListRecPtr;//recursive version

		std::span<const CompPlugin> plugins;

		std::span<const std::string_view> rootPaths;//All of them must have some crate.

		//will use exactly that many threads, not any less, not any more
		size_t extraThreadCount = 
			(size_t)std::max(//max, so no negative thread counts
				int64_t(std::thread::hardware_concurrency()) 
					- 3,//2 sys threads, 1 for main thread
				0LL
		);
	};
	//Could repr some js file, some wasm blob, a jar / class, or even some exe / dll.
	struct CompEntryPoint
	{
		std::string entryPointFile;//path to file that defined this entry-point, or empty
		std::string fileName;
		std::vector<uint8_t> contents;
	};
	struct CompOutput
	{
		std::vector<CompEntryPoint> entryPoints;
		//TODO: info for lock file appending?
	};
	inline CompOutput compile(const CompCfg& cfg)
	{

	}
}