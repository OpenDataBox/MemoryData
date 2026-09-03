package com.linkpoket.source.domain.aisearch.service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import com.linkpoket.source.domain.aisearch.dto.response.AiSearchSourceDto;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;
import com.linkpoket.source.domain.aisearch.entity.AiItemBody;
import com.linkpoket.source.domain.aisearch.entity.ExtractionStatus;
import com.linkpoket.source.domain.aisearch.repository.AiFolderMetaDataAccess;
import com.linkpoket.source.domain.aisearch.repository.AiItemBodyDataAccess;

import com.linkpoket.source.domain.page.repository.PageDataAccess;
import com.linkpoket.source.domain.page.entity.Page;
import com.linkpoket.source.domain.folder.repository.FolderDataAccess;
import com.linkpoket.source.domain.folder.entity.Folder;
import com.linkpoket.source.domain.postit.repository.FolderPostItDataAccess;
import com.linkpoket.source.domain.postit.entity.FolderPostIt;
import com.linkpoket.source.domain.link.repository.LinkDataAccess;
import com.linkpoket.source.domain.note.repository.NoteDataAccess;
import com.linkpoket.source.domain.file.repository.FileDataAccess;
import com.linkpoket.source.domain.file.entity.File;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 실시간 검색 파이프라인에서 페이지, 폴더, 아이템의 엔티티 상세 정보를 조회하여
 * LLM 컨텍스트 주입용 마크다운 형식으로 가공 및 조합하는 컴포저 서비스 클래스입니다.
 */
@Component
@RequiredArgsConstructor
@Slf4j
@Transactional(readOnly = true)
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class MdComposer {

	static final String USER_DATA_START = "<<<USER_DATA_START>>>\n";
	static final String USER_DATA_END = "\n<<<USER_DATA_END>>>";

	private static final int MAX_FOLDERS = 50;
	private static final int MAX_ITEMS = 100;
	private static final int APPROX_CHARS_PER_TOKEN = 2;

	private final PageDataAccess pageDataAccess;
	private final FolderDataAccess folderDataAccess;
	private final AiFolderMetaDataAccess folderMetaDataAccess;
	private final FolderPostItDataAccess folderPostItDataAccess;
	private final LinkDataAccess linkDataAccess;
	private final NoteDataAccess noteDataAccess;
	private final FileDataAccess fileDataAccess;
	private final AiItemBodyDataAccess itemBodyDataAccess;

	@Value("${ai-search.top-n-items:10}")
	private int topNItems;

	@Value("${ai-search.max-body-tokens:8000}")
	private int maxBodyTokens;

	/**
	 * 1단계: 회원이 접근 가능한 페이지 ID 목록을 마크다운 형태의 프롬프트 텍스트로 전환합니다.
	 *
	 * @param accessiblePageIds 접근 가능한 페이지 식별자 목록
	 * @return 마크다운으로 조립된 페이지 목록 텍스트
	 */
	public String composePageList(final List<Long> accessiblePageIds) {
		if (accessiblePageIds == null || accessiblePageIds.isEmpty()) {
			return USER_DATA_START + "(접근 가능한 페이지가 없습니다)" + USER_DATA_END;
		}
		final StringBuilder builder = new StringBuilder("### 접근 가능 페이지 목록\n");
		for (final Long pageId : accessiblePageIds) {
			pageDataAccess.findById(pageId).ifPresent(page -> {
				builder.append("- 페이지 ID: ").append(page.getId())
					.append(", 제목: ").append(page.getTitle()).append("\n");
			});
		}
		return USER_DATA_START + builder.toString() + USER_DATA_END;
	}

	/**
	 * 2단계: 회원이 접근 가능한 폴더의 상세 구조 및 AI 요약 설명을 마크다운 프롬프트로 가공합니다.
	 *
	 * @param accessiblePageIds 접근 권한이 부여된 상위 페이지 식별자 목록
	 * @return 폴더 프롬프트 및 탐색 대상 폴더 ID 리스트를 내포하는 PromptBundle 객체
	 */
	public PromptBundle composeFolderListPrompt(final List<Long> accessiblePageIds) {
		final List<Folder> folders = new ArrayList<>();
		for (final Long pageId : accessiblePageIds) {
			folders.addAll(folderDataAccess.findTopLevelFolders(pageId));
		}
		final List<Long> parentIds = folders.stream().map(Folder::getId).collect(Collectors.toList());
		if (!parentIds.isEmpty()) {
			folders.addAll(folderDataAccess.findChildFolders(parentIds));
		}

		final List<Folder> limitFolders = folders.stream().limit(MAX_FOLDERS).collect(Collectors.toList());
		final List<Long> folderIds = limitFolders.stream().map(Folder::getId).collect(Collectors.toList());
		final List<AiFolderMeta> metas = folderMetaDataAccess.findAllByFolderIdIn(folderIds);
		final Map<Long, String> metaMap = metas.stream()
			.collect(Collectors.toMap(AiFolderMeta::getFolderId, AiFolderMeta::getDescription));

		final StringBuilder builder = new StringBuilder("### 접근 가능 폴더 목록\n");
		for (final Folder folder : limitFolders) {
			final String desc = metaMap.getOrDefault(folder.getId(), "");
			final Long parentId = folder.getParentFolder() != null ? folder.getParentFolder().getId() : null;
			builder.append("- 폴더 ID: ").append(folder.getId())
				.append(", 상위 페이지 ID: ").append(folder.getPageId())
				.append(", 상위 폴더 ID: ").append(parentId != null ? parentId : "NULL")
				.append(", 제목: ").append(folder.getName())
				.append(", 설명 요약: ").append(desc).append("\n");
		}

		return new PromptBundle(USER_DATA_START + builder.toString() + USER_DATA_END, folderIds);
	}

	/**
	 * 3단계: 후보군으로 꼽힌 폴더들에 적재된 하위 아이템(링크, 노트, 파일 등)들을 로드하고,
	 * 정렬 및 가공하여 LLM 질의용 최종 마크다운 프롬프트를 생성합니다.
	 *
	 * @param candidateFolderIds LLM에 의해 선정된 후보 폴더 식별자 목록
	 * @return 아이템 목록 프롬프트 및 검색 소스 리스트가 수록된 ItemBodyBundle 객체
	 */
	public ItemBodyBundle composeItemsPrompt(final List<Long> candidateFolderIds) {
		final List<ItemView> items = new ArrayList<>();
		for (final Long folderId : candidateFolderIds) {
			linkDataAccess.findAllByFolderIdIn(List.of(folderId)).forEach(l -> {
				final String fallback = (nullToEmpty(l.getDescription())
					+ (l.getProviderName() != null ? " (출처: " + l.getProviderName() + ")" : "")).trim();
				items.add(new ItemView(l.getId(), "LINK", l.getName(), l.getUrl(), fallback, l.getEditedAt()));
			});
			noteDataAccess.findAllByFolderIdIn(List.of(folderId)).forEach(n ->
				items.add(new ItemView(n.getId(), "NOTE", n.getName(), "", n.getContent(), n.getEditedAt()))
			);
			fileDataAccess.findFileEntitiesByFolderIds(List.of(folderId)).forEach(f -> {
				final String fallback = "(파일 본문 미확보 — 파일명만 제공됨)";
				items.add(new ItemView(f.getId(), "FILE", f.getName(), f.getFileUrl(), fallback, f.getEditedAt()));
			});
		}

		final List<ItemView> sorted = items.stream()
			.sorted(Comparator.comparing(ItemView::editedAt, Comparator.nullsLast(Comparator.reverseOrder())))
			.limit(MAX_ITEMS)
			.collect(Collectors.toList());

		final List<Long> itemIds = sorted.stream().map(ItemView::itemId).collect(Collectors.toList());
		final Map<Long, String> bodyMap = itemBodyDataAccess.findAllByItemIdIn(itemIds).stream()
			.filter(b -> b.getExtractionStatus() == ExtractionStatus.DONE)
			.filter(b -> b.getBodyText() != null && !b.getBodyText().isBlank())
			.collect(Collectors.toMap(AiItemBody::getItemId, AiItemBody::getBodyText));

		final List<AiSearchSourceDto> sources = new ArrayList<>();
		final Map<Long, String> itemLines = new java.util.LinkedHashMap<>();
		final StringBuilder builder = new StringBuilder("### 관련 아이템 목록\n");
		int currentTokens = 0;

		for (final ItemView item : sorted) {
			// 원문 추출(AiItemBody)이 있으면 그걸 우선 사용하고, 없으면 ItemView 생성 시 채운 폴백(NOTE 원문 / LINK 설명·출처 / FILE 안내문)을 쓴다.
			// 링크·파일이 추출 전이라도 빈 문자열이 나가는 일이 없도록 한다.
			final String textBody = bodyMap.getOrDefault(item.itemId(), item.body());
			final int bodyLen = textBody != null ? textBody.length() : 0;
			final int bodyEstTokens = bodyLen / APPROX_CHARS_PER_TOKEN;

			if (currentTokens + bodyEstTokens > maxBodyTokens && sources.size() >= topNItems) {
				break;
			}

			sources.add(new AiSearchSourceDto(item.itemId(), item.itemType(), item.title(), item.url()));
			currentTokens += bodyEstTokens;

			final String line = "- 아이템 ID: " + item.itemId()
				+ ", 타입: " + item.itemType()
				+ ", 제목: " + item.title()
				+ ", 본문 텍스트: " + textBody + "\n";
			itemLines.put(item.itemId(), line);
			builder.append(line);

			if (sources.size() >= topNItems) {
				break;
			}
		}

		return new ItemBodyBundle(USER_DATA_START + builder.toString() + USER_DATA_END, sources, itemLines);
	}

	private String nullToEmpty(final String value) {
		return value == null ? "" : value;
	}
}
