package com.linkpoket.source.domain.aisearch.event;

import java.util.Optional;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.event.EventListener;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.orm.ObjectOptimisticLockingFailureException;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import com.linkpoket.source.domain.aisearch.entity.AiPageMeta;
import com.linkpoket.source.domain.aisearch.error.AiSearchErrorCode;
import com.linkpoket.source.domain.aisearch.repository.AiPageMetaDataAccess;
import com.linkpoket.source.domain.aisearch.service.PageDescriptionBuilder;
import com.linkpoket.source.domain.aisearch.service.PageDescriptionMismatchChecker;
import com.linkpoket.source.global.config.AsyncConfig;
import com.linkpoket.source.global.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 특정 폴더의 요약 변경에 맞춰 발행된 PageDescriptionCheckEvent를 수신하여
 * 상위 페이지 레벨의 설명 적합성을 판단 및 조건부로 페이지 요약을 갱신하는 비동기 이벤트 리스너 클래스입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class PageDescriptionUpdateListener {

	private final AiPageMetaDataAccess pageMetaDataAccess;
	private final PageDescriptionMismatchChecker mismatchChecker;
	private final PageDescriptionBuilder descriptionBuilder;

	/**
	 * 페이지 설명 갱신 체크 이벤트를 비동기로 호출 수신하여 요약 갱신 여부를 판별하고 페이지 요약을 조건부로 새로 씁니다.
	 *
	 * @param event 대상 페이지와 갱신 원인이 된 폴더 식별 정보를 갖는 이벤트
	 */
	@Async(AsyncConfig.AI_SEARCH_TASK_EXECUTOR)
	@EventListener
	@Transactional
	public void handleFolderDescriptionChanged(final PageDescriptionCheckEvent event) {
		final Long pageId = event.pageId();
		final AiPageMeta meta = pageMetaDataAccess.findByPageId(pageId).orElse(null);
		final String currentDescription = meta != null ? meta.getDescription() : null;

		final boolean needsRebuild = currentDescription == null || currentDescription.isBlank()
			|| mismatchChecker.isMismatch(pageId, currentDescription, event.folderId());

		if (!needsRebuild) {
			log.debug("[AiSearch] page description kept as-is. pageId={}", pageId);
			return;
		}
		final String rebuilt = descriptionBuilder.build(pageId);
		upsertPageDescription(pageId, rebuilt);
		log.info("[AiSearch] page description rebuilt. pageId={}", pageId);
	}

	/**
	 * 페이지 요약 설명문을 데이터베이스에 저장 갱신합니다. PK 충돌 레이스 컨디션을 대비하여 안전한 예외 복구 및 재시도 저장 흐름을 갖춥니다.
	 *
	 * @param pageId  페이지 식별자
	 * @param rebuilt 갱신할 페이지 설명 텍스트 내용
	 */
	private void upsertPageDescription(final Long pageId, final String rebuilt) {
		final String safeDescription = rebuilt == null ? "" : rebuilt;
		try {
			final AiPageMeta meta = pageMetaDataAccess.findByPageId(pageId)
				.orElseGet(() -> AiPageMeta.builder()
					.pageId(pageId)
					.description(safeDescription)
					.build());
			meta.updateDescriptionAndEmbedding(safeDescription, meta.getEmbedding());
			pageMetaDataAccess.saveAndFlush(meta);
		} catch (final DataIntegrityViolationException | ObjectOptimisticLockingFailureException e) {
			final AiPageMeta meta = pageMetaDataAccess.findByPageId(pageId)
				.orElseThrow(() -> new BusinessException(AiSearchErrorCode.SEARCH_UNAVAILABLE));
			meta.updateDescriptionAndEmbedding(safeDescription, meta.getEmbedding());
			pageMetaDataAccess.save(meta);
		}
	}
}